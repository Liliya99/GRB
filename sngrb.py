import numpy as np
from scipy.optimize import least_squares, curve_fit, fmin
from scipy.stats.mstats import theilslopes
from scipy.stats import skewnorm, norm, multivariate_normal
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rc, rcParams
import corner
from joblib import Parallel, delayed, cpu_count
#from uncertainties import wrap as uw
from numba import jit, vectorize
import warnings
import json
from tqdm import tqdm
warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore', over='ignore')

#Назначаем константы:
pi = 4 * np.arctan(1.0)
c = 299792.458 #В километрах в секунду
H0 = 70

Mpc_in_cm = 3.08567758e24

err_ql = norm.cdf(-1.0)
err_qu = norm.cdf(1.0)

paper_linewidth = 3.37689 #Ширина колонки текста в MNRAS-овском шаблоне
paper_textwidth = 7.03058 #Ширина страницы в MNRAS-овском шаблоне

#Наcтройка шрифтов в matplotlib:
rc('font', family = 'Times New Roman')
rc('font', size=10)
rc('text', usetex=False)
rc('text.latex',preamble=r'\usepackage[utf8]{inputenc}')
rc('text.latex',preamble=r'\usepackage[russian]{babel}')
rc('mathtext', fontset='stix')
rc('figure', figsize = (5, 4.25))

#Важный параметр -- размер выборок Монте-Карло:
#Для тестирования можно брать значение 1024 -- с ним код прогоняется недолго
#Для окончательной прогонки лучше брать значения побольше -- например, 10000
sample_size = 1000

#Количество параллельных процессов:
Nthreads = cpu_count()

#==============================================================================
#    Далее идёт часть кода, в которой я определяю различные нужные функции
#==============================================================================

#Обычная линейная функция (нужна для фиттинга)
def lin(x, A, B):
    return A * x + B

#Функция, возвращающая верхнюю и нижнюю 1-сигма ограничивающую кривую для линейной функции
def lin_band(xs, a_sample, b_sample, conf_int=100 * (err_qu - err_ql)):
    upper_bord = np.vectorize(lambda x: np.percentile(lin(x, a_sample, b_sample), (100 + conf_int) / 2))(xs)
    lower_bord = np.vectorize(lambda x: np.percentile(lin(x, a_sample, b_sample), (100 - conf_int) / 2))(xs)
    return (lower_bord, upper_bord)

#Тоже линейная функция, но с переставленными x и y
def lin_inv(y, A, B):
    return (y - B) / A

def split_normal_cdf(x: float, mu: float, lower_sigma: float, upper_sigma: float):
    if x < mu:
        return norm.cdf(x, mu, lower_sigma)
    else:
        return norm.cdf(x, mu, upper_sigma)

#Создание выборки из разделённо-нормального распределения
def random_split_normal(mu: float, lower_sigma: float, upper_sigma: float, amount: int = 1024):
    z = np.random.normal(0, 1, amount)
    return mu + z * (- lower_sigma * (np.sign(z) - 1) / 2 + upper_sigma * (np.sign(z) + 1) / 2)

#Создание выборки из логарифмического разделённо-нормального распределения
def random_log_split_normal(mu: float, lower_sigma: float, upper_sigma: float, amount: int = 1024):
    return 10 ** (random_split_normal(np.log10(mu), np.log10(mu / (mu - lower_sigma)), np.log10((mu + upper_sigma) / mu), amount = amount))

@jit
def twopiece_normal_ppf(q: float, mu: float, lower_sigma: float, upper_sigma: float):
    if q < lower_sigma / (lower_sigma + upper_sigma):
        return norm.ppf(q * (lower_sigma + upper_sigma) / 2.0 / lower_sigma, mu, lower_sigma)
    else:
        return norm.ppf((q * (lower_sigma + upper_sigma) - lower_sigma + upper_sigma) / 2.0 / upper_sigma, mu, upper_sigma)

@vectorize(['float64(float64,float64,float64,float64)'])
def twopiece_normal_ppf_ufunc(q, mu, lsigma, usigma):
    return twopiece_normal_ppf(q, mu, lsigma, usigma)

def random_twopiece_normal(mu: float, lower_sigma: float, upper_sigma: float, amount: int = 1024):
    return np.vectorize(twopiece_normal_ppf)(np.random.uniform(size=amount), mu, lower_sigma, upper_sigma)

def random_offset_lognormal(mu: float, lower_sigma: float, upper_sigma: float, amount: int = 1024):
    if lower_sigma == upper_sigma:
        return np.random.normal(mu, upper_sigma, size=amount)
    elif lower_sigma < upper_sigma:
        delta = mu - upper_sigma * lower_sigma / (upper_sigma - lower_sigma)
        return 10 ** np.random.normal(np.log10(mu - delta), np.log10((mu + upper_sigma - delta) / (mu - delta)), size=amount) + delta
    else:
        delta = mu - lower_sigma * upper_sigma / (lower_sigma - upper_sigma)
        return - 10 ** np.random.normal(np.log10(mu - delta), np.log10((mu + lower_sigma - delta) / (mu - delta)), size=amount) - delta + 2 * mu

def offset_lognorm_cdf(x, mu, lower_sigma, upper_sigma):
    if lower_sigma < upper_sigma:
        delta = mu - lower_sigma * upper_sigma / (upper_sigma - lower_sigma)
        return norm.cdf(np.log10(x-delta), np.log10(mu-delta), np.log10((mu+upper_sigma-delta) / (mu-delta)))
    elif lower_sigma > upper_sigma:
        delta = mu - lower_sigma * upper_sigma / (lower_sigma - upper_sigma)
        return 1 - norm.cdf(np.log10(-x+2*mu-delta), np.log10(mu-delta), np.log10((mu+lower_sigma-delta) / (mu-delta)))
    else:
        return norm.cdf(x, mu, upper_sigma)

def offset_lognorm_ppf(q, mu, lower_sigma, upper_sigma):
    if lower_sigma < upper_sigma:
        delta = mu - lower_sigma * upper_sigma / (upper_sigma - lower_sigma)
        return 10 ** norm.ppf(q, np.log10(mu-delta), np.log10((mu+upper_sigma-delta) / (mu-delta))) + delta
    elif lower_sigma > upper_sigma:
        delta = mu - lower_sigma * upper_sigma / (lower_sigma - upper_sigma)
        return - 10 ** norm.ppf(1-q, np.log10(mu-delta), np.log10((mu+lower_sigma-delta) / (mu-delta))) - delta + 2 * mu
    else:
        return norm.ppf(q, mu, upper_sigma)

def offset_lognorm_pdf(x, mu, lower_sigma, upper_sigma):
    if lower_sigma < upper_sigma:
        delta = mu - lower_sigma * upper_sigma / (upper_sigma - lower_sigma)
        return norm.pdf(np.log10(x-delta), np.log10(mu-delta), np.log10((mu+upper_sigma-delta) / (mu-delta))) / (x - delta) / np.log(10)
    elif lower_sigma > upper_sigma:
        delta = mu - lower_sigma * upper_sigma / (lower_sigma - upper_sigma)
        return norm.pdf(np.log10(-x+2*mu-delta), np.log10(mu-delta), np.log10((mu+lower_sigma-delta) / (mu-delta))) / (2 * mu - x - delta) / np.log(10)
    else:
        return norm.pdf(x, mu, upper_sigma)

def smooth_split_normal_ppf(q, mu, lower_sigma, upper_sigma):
    if (q > norm.cdf(-2.0)) and (q < 0.5):
        x = (q - 0.5) / (0.5 - norm.cdf(-2.0))
        w = (np.cos(x * pi) + 1) / 2.0
        return offset_lognorm_ppf(q, mu, lower_sigma, upper_sigma) * w + norm.ppf(q, mu, lower_sigma) * (1 - w)
    elif (q >= 0.5) and (q < norm.cdf(2.0)):
        x = (q - 0.5) / (norm.cdf(2.0) - 0.5)
        w = (np.cos(x * pi) + 1) / 2.0
        return offset_lognorm_ppf(q, mu, lower_sigma, upper_sigma) * w + norm.ppf(q, mu, upper_sigma) * (1 - w)
    elif q <= norm.cdf(-2.0):
        return norm.ppf(q, mu, lower_sigma)
    else:
        return norm.ppf(q, mu, upper_sigma)

def random_smooth_split_normal(mu, lower_sigma, upper_sigma, amount=1024):
    return np.vectorize(smooth_split_normal_ppf)(np.random.uniform(size=amount), mu, lower_sigma, upper_sigma)

#Задание констант: число промежутков интегрирования
intit_dist = 10001
intit_sp = 10001
intit_sperr = 10001

#Далее идёт задание космологических функций с jit-компиляцией
@jit(nopython=True)
def r(z, pars_cosm, am=intit_dist):
    H0, Omm, OmDE, Omk, w = pars_cosm
    z_int_range = np.linspace(0, z, am)
    h_int_range = 1 / h(z_int_range, pars_cosm)
    z_int = np.trapz(h_int_range, z_int_range)
    return c / H0 * z_int

@jit(nopython=True)
def h(z, pars_cosm):
    H0, Omm, OmDE, Omk, w = pars_cosm
    return np.sqrt(Omm * (1 + z) ** 3 + OmDE * (1 + z) ** (3 * (1 + w)) - Omk * (1 + z) ** 2)

@jit(nopython=True)
def l(z, pars_cosm, am=intit_dist):
    H0, Omm, OmDE, Omk, w = pars_cosm
    z_int_range = np.linspace(0, z, am)
    h_int_range = 1 / h(z_int_range, pars_cosm)
    z_int = np.trapz(h_int_range, z_int_range)
    if Omk == 0:
        return c / H0 * z_int
    elif Omk > 0:
        return c / H0 / np.sqrt(np.abs(Omk)) * np.sin(np.sqrt(np.abs(Omk)) * z_int)
    else:
        return c / H0 / np.sqrt(np.abs(Omk)) * np.sinh(np.sqrt(np.abs(Omk)) * z_int)
    
@jit(nopython=True)
def dl(z, pars_cosm):
    return (1 + z) * l(z, pars_cosm)

@jit(nopython=True)
def de(z, pars_cosm):
    return np.sqrt(1 + z) * l(z, pars_cosm)

@jit(nopython=True)
def mu(z, pars_cosm):
    return 25 + 5 * np.log10(dl(z, pars_cosm))

#@vectorize(['float64(float64)'])

@jit(nopython=True)
def EN(E, pars_spec):
    alpha, E_p = pars_spec
    E_0 = E_p / (2 + alpha)
    return E ** (alpha + 1) * np.exp(-E / E_0)

@jit(nopython=True)
def int_E(lims, pars_spec, am=intit_sp):
    E_range = np.linspace(lims[0], lims[1], am)
    EN_range = EN(E_range, pars_spec)
    return np.trapz(EN_range, E_range)

@jit(nopython=True)
def S_bolo(z, alpha, E_p, S_obs):
    denom_int = int_E((15, 150), (alpha, E_p))
    num_int = int_E((1 / (1 + z), 1e4 / (1 + z)), (alpha, E_p))
    if denom_int == 0:
        return np.inf
    else:
        return S_obs * num_int / denom_int * 1e-7
    
@jit(nopython=True)
def S_bolo_corr(z, alpha, E_p, S_obs, k):
    S_bolo_inner = S_bolo(z, alpha, E_p, S_obs)
    return S_bolo_inner / (1 + z) ** k

@jit(nopython=True)
def E_iso(z, alpha, E_p, S_obs, pars_cosm):
    return 4 * pi * (de(z, pars_cosm) * Mpc_in_cm) ** 2 * S_bolo(z, alpha, E_p, S_obs)

@jit(nopython=True)
def E_iso_corr(z, alpha, E_p, S_obs, pars_cosm, k):
    return 4 * pi * (de(z, pars_cosm) * Mpc_in_cm) ** 2 * S_bolo_corr(z, alpha, E_p, S_obs, k)

@jit(nopython=True)
def mu_A(z, alpha, E_p, S_obs, a, b, k_par):
    S_bolo_arg = S_bolo(z, alpha, E_p, S_obs)
    if S_bolo_arg == 0 or S_bolo_arg > 1e10:
        return 0
    else:
        return 25 + 2.5 * (np.log10((1 + z) ** (k_par + 1) / (4 * pi * S_bolo_arg * Mpc_in_cm ** 2)) + a * np.log10(E_p * (1 + z)) + b)

@vectorize(['float64(float64,float64,float64,float64,float64,float64,float64)'])
def mu_A_ufunc(z, alpha, E_p, S_obs, a, b, k_par):
    return mu_A(z, alpha, E_p, S_obs, a, b, k_par)

@jit(nopython=True)
def Amx(z, E_p):
    return np.log10(E_p * (1 + z))

@jit(nopython=True)
def Amy(z, alpha, E_p, S_obs, pars_cosm):
    return np.log10(E_iso(z, alpha, E_p, S_obs, pars_cosm))

#uE_iso = uw(E_iso)
#umu_A = uw(mu_A)
#uAmx = uw(Amx)
#uAmy = uw(Amy)

#Функция, создающая выборки для alpha, Ep и Sobs
def make_samples_split(ii, data, amount=1024):
    alpha_arg, Dalpha_arg_d, Dalpha_arg_u, E_p_arg, DE_p_arg_d, DE_p_arg_u, S_obs_arg, DS_obs_arg, z_arg = data
    alpha_sample = random_split_normal(alpha_arg[ii], Dalpha_arg_d[ii], Dalpha_arg_u[ii], amount)
    E_p_sample = random_log_split_normal(E_p_arg[ii], DE_p_arg_d[ii], DE_p_arg_u[ii], amount=amount)
    #E_p_sample[E_p_sample <= 1e-3] = 1e-3
    S_obs_sample = np.random.normal(S_obs_arg[ii], DS_obs_arg[ii], amount)
    return (alpha_sample, E_p_sample, S_obs_sample)

#Смещённое логарифмическое распределение:
def make_samples(ii, data, amount=1024):
    alpha_arg, Dalpha_arg_d, Dalpha_arg_u, E_p_arg, DE_p_arg_d, DE_p_arg_u, S_obs_arg, DS_obs_arg, z_arg = data
    alpha_sample = random_smooth_split_normal(alpha_arg[ii], Dalpha_arg_d[ii], Dalpha_arg_u[ii], amount)
    E_p_sample = 10 ** (random_smooth_split_normal(np.log10(E_p_arg[ii]), np.log10(E_p_arg[ii] / (E_p_arg[ii] - DE_p_arg_d[ii])), np.log10((E_p_arg[ii] + DE_p_arg_u[ii]) / E_p_arg[ii]), amount = amount))
    S_obs_sample = np.random.normal(S_obs_arg[ii], DS_obs_arg[ii], amount)
    return (alpha_sample, E_p_sample, S_obs_sample)

#Функция, возвращающая медианы и пределы доверительного интервала для выборок
def get_meds_and_lims(samples):
    N_inner = np.shape(samples)[0]
    meds = np.empty(N_inner)
    dlim = np.empty(N_inner)
    ulim = np.empty(N_inner)
    for i in np.arange(N_inner):
        meds[i] = np.median(samples[i,:][np.isfinite(samples[i,:])])
        dlim[i] = meds[i] - np.percentile(samples[i,:][np.isfinite(samples[i,:])], 100 * err_ql)
        ulim[i] = np.percentile(samples[i,:][np.isfinite(samples[i,:])], 100 * err_qu) - meds[i]
    return (meds, dlim, ulim)

def twopiece_lhood(mu, data):
    data_arg = data[np.isfinite(data)]
    return - np.sum((data_arg[data_arg <= mu] - mu) ** 2) ** (1 / 3) - np.sum((data_arg[data_arg > mu] - mu) ** 2) ** (1 / 3)

def get_twopiece_pars(samples):
    if np.size(np.shape(samples)) == 1:
        mu_est = fmin(lambda mu_arg: - twopiece_lhood(mu_arg, samples), 0.0, disp=False)[0]
        sample_cl = samples[np.isfinite(samples)]
        L_est = twopiece_lhood(mu_est, sample_cl)
        s1_est = np.sqrt(- L_est / np.size(sample_cl) * (np.sum((sample_cl[sample_cl <= mu_est] - mu_est) ** 2)) ** (2 / 3))
        s2_est = np.sqrt(- L_est / np.size(sample_cl) * (np.sum((sample_cl[sample_cl > mu_est] - mu_est) ** 2)) ** (2 / 3))
        return (mu_est, s1_est, s2_est)
    else:
        N_inner = np.shape(samples)[0]
        mu_ests = np.empty(N_inner)
        s1_ests = np.empty(N_inner)
        s2_ests = np.empty(N_inner)
        for i in np.arange(N_inner):
            mu_ests[i] = fmin(lambda mu_arg: -twopiece_lhood(mu_arg, samples[i,:]), 0.0, disp=False)[0]
            sample_cl = samples[i,:][np.isfinite(samples[i,:])]
            L_est = twopiece_lhood(mu_ests[i], sample_cl)
            s1_ests[i] = np.sqrt(- L_est / np.size(sample_cl) * (np.sum((sample_cl[sample_cl <= mu_ests[i]] - mu_ests[i]) ** 2)) ** (2 / 3))
            s2_ests[i] = np.sqrt(- L_est / np.size(sample_cl) * (np.sum((sample_cl[sample_cl > mu_ests[i]] - mu_ests[i]) ** 2)) ** (2 / 3))
        return (mu_ests, s1_ests, s2_ests)

#Функция, вычисляющая x и y в плоскости Амати для выборок
def sample_Amatixy(ii, data, pars_cosm):
    alpha_sample, E_p_sample, S_obs_sample, z_arg = data
    Amx_sample = Amx(z_arg[ii], E_p_sample)
    Amx_sample[np.isnan(Amx_sample)] = -np.inf
    Amy_sample = np.vectorize(lambda z, alpha, E_p, S_obs: Amy(z, alpha, E_p, S_obs, pars_cosm))(z_arg[ii], alpha_sample, E_p_sample, S_obs_sample)
    Amy_sample[np.isnan(Amy_sample)] = np.inf
    return (Amx_sample, Amy_sample)

#Функция, вычисляющая Sbolo для выборок
def sample_S_bolo(ii, data):
    alpha_sample, E_p_sample, S_obs_sample, z_arg = data
    S_bolo_sample = S_bolo(z_arg[ii], alpha_sample, E_p_sample, S_obs_sample) * Mpc_in_cm ** 2
    return S_bolo_sample

#Функция, вычисляющая mu_A для выборок
def sample_mu_A(ii, data, a, b, k):
    alpha_sample, E_p_sample, S_obs_sample, z_arg = data
    mu_A_sample = np.vectorize(mu_A)(z_arg[ii], alpha_sample, E_p_sample, S_obs_sample, a, b, k)
    return mu_A_sample

#Отрисовка красивых графиков в плоскости Амати
def plot_Amati(fig, ax, data, imagename, legendname, plot_errbars=True, plot_samples=True, xlim = (0.75, 4.25), ylim = (49, 55), col1='darkgreen', col2='forestgreen', cmap='Greens'):
    if plot_errbars and plot_samples:
        a_sample, b_sample, Amx_flat, Amy_flat, Amx_meds, Amy_meds, Amx_ulim, Amy_ulim, Amx_dlim, Amy_dlim = data
    elif plot_errbars:
        a_sample, b_sample, Amx_meds, Amy_meds, Amx_ulim, Amy_ulim, Amx_dlim, Amy_dlim = data
    elif plot_samples:
        a_sample, b_sample, Amx_flat, Amy_flat, Amx_meds, Amy_meds = data
    a_est = np.median(a_sample)
    b_est = np.median(b_sample)
    Da_est = (np.percentile(a_sample, 100 * err_qu) - np.percentile(a_sample, 100 * err_ql)) / 2
    Db_est = (np.percentile(b_sample, 100 * err_qu) - np.percentile(b_sample, 100 * err_ql)) / 2
    x_range = np.linspace(xlim[0], xlim[1], 100)
    y_range = lin(x_range, a_est, b_est)
    y_range_1sig_d, y_range_1sig_u = lin_band(x_range, a_sample, b_sample)
    Am_res_sigma = (Amy_meds - np.median(a_sample) * Amx_meds - np.median(b_sample)).std()
    ax.plot(x_range, y_range, c='k', zorder=10)
    ax.plot([], [], c='grey', linestyle=':', linewidth=0.75)
    ax.plot(x_range, y_range + Am_res_sigma, c='k', zorder=10, linestyle='dashed', linewidth=1)
    ax.fill_between(x_range, y_range_1sig_d, y_range_1sig_u, alpha=0.2, color='k', linestyle=':', linewidth=0.75)
    ax.plot(x_range, y_range - Am_res_sigma, c='k', zorder=10, linestyle='dashed', linewidth=1)
    if plot_errbars:
        ax.errorbar(Amx_meds, Amy_meds, yerr=np.array([Amy_dlim, Amy_ulim]), xerr=np.array([Amx_dlim, Amx_ulim]), linestyle='', linewidth=0.3, marker='o', markersize=1.15, color=col1, rasterized=True)
    if plot_samples:
        limmask = (Amx_flat > xlim[0]) * (Amx_flat < xlim[1]) * (Amy_flat > ylim[0]) * (Amy_flat < ylim[1])
        ax.scatter(Amx_flat[limmask], Amy_flat[limmask], s=0.0007 * (1000 / sample_size) ** 1.5, c=col2, marker=".", rasterized=True)
        #ax.scatter_density(Amx_flat[limmask], Amy_flat[limmask], color=col2, vmin=0, vmax=50)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_title('Amati plane, ' + legendname)
    ax.set_xlabel(r'$\mathrm{log}\,(E_{\mathrm{p,i}} \,/\, 1\,\mathrm{keV})$')
    ax.set_ylabel(r'$\mathrm{log}\,(E_{\mathrm{iso}}\,/\, 1\,\mathrm{erg})$')
    ax.legend([ '$a=' + str(np.around(a_est, 2)) + '\\pm' + str(np.around(Da_est, 2)) + '$,\n$b=' + str(np.around(b_est, 1)) + '\\pm' + str(np.around(Db_est, 1)) + '$', '$1\\sigma$-conf. region', '$1\\sigma$-pred. band'], loc=4)
    fig.tight_layout()
    if plot_errbars and plot_samples:
        fig.savefig('pics/' + imagename + '_errbars_and_samples.pdf', dpi=225)
        fig.savefig('pics/' + imagename + '_errbars_and_samples.png', dpi=225)
    elif plot_errbars:
        fig.savefig('pics/' + imagename + '_errbars.pdf', dpi=225)
        fig.savefig('pics/' + imagename + '_errbars.png', dpi=225)
    elif plot_samples:
        fig.savefig('pics/' + imagename + '_samples.pdf', dpi=225)
        fig.savefig('pics/' + imagename + '_samples.png', dpi=225)

#==============================================================================
#               Далее идёт непосредственно скриптовая часть
#==============================================================================

#Загрузим каталог:
cat0 = pd.read_excel('./catalogues/work_catalog_2022.xlsx', usecols = 'A,C:H,M:O,X')
cat0 = cat0.replace('N', np.nan)
cat0 = cat0.replace(0, np.nan)
cat0 = cat0.replace('>100', 100)
cat0 = cat0.replace('>300', 300)

#Заменим неизвестные ошибки медианными:
def replace_nan_by_med(cat, colnames, sym=False):
    if sym:
        col, err = colnames
        #medval = np.median(cat[col][~np.isnan(cat[col])])
        mederr = np.median(cat[err][~np.isnan(cat[err])] / cat[col][~np.isnan(cat[err])])
        #cat.loc[np.isnan(cat[col]), col] = medval
        cat.loc[np.isnan(cat[err]), err] = mederr * cat[col][np.isnan(cat[err])]
    else:
        col, toplim, botlim = colnames
        toperr = ( cat[toplim][~np.isnan(cat[toplim])] - cat[col][~np.isnan(cat[toplim])] ) / cat[col][~np.isnan(cat[toplim])]
        boterr = ( cat[col][~np.isnan(cat[botlim])] - cat[botlim][~np.isnan(cat[botlim])] ) / cat[col][~np.isnan(cat[botlim])]
        #valmed = np.median(cat[col][~np.isnan(cat[col])])
        topmed = np.median(toperr)
        botmed = np.median(boterr)
        #cat.loc[np.isnan(cat[col]), col] = valmed
        cat.loc[np.isnan(cat[toplim]), toplim] = cat[col][np.isnan(cat[toplim])] * (1 + topmed)
        cat.loc[np.isnan(cat[botlim]), botlim] = cat[col][np.isnan(cat[botlim])] * (1 - botmed)

replace_nan_by_med(cat0, ('CPL:alpha', 'CPL:alpha+', 'CPL:alpha-'))
replace_nan_by_med(cat0, ('CPL:Ep', 'CPL:Ep+', 'CPL:Ep-'))
replace_nan_by_med(cat0, ('BAT Fluence\n(15-150 keV)\n[10-7 erg/cm2]', 'BAT Fluence\n90% Error\n(15-150 keV)\n[10-7 erg/cm2]'), sym=True)

#Найдём значения E_iso и E_p,i
#Сначала считаем наши наблюдаемые параметры и их погрешности
S_obs_arr = np.array(cat0['BAT Fluence\n(15-150 keV)\n[10-7 erg/cm2]'])
E_p_arr = np.array(cat0['CPL:Ep'])
E_p_arr_u = np.array(cat0['CPL:Ep+'])
nonans_mask = ~np.isnan(S_obs_arr) * (E_p_arr_u - E_p_arr > 0.0)
S_obs_arr = S_obs_arr[nonans_mask]
E_p_arr = E_p_arr[nonans_mask]
E_p_arr_u = E_p_arr_u[nonans_mask]
E_p_arr_d = np.array(cat0['CPL:Ep-'])[nonans_mask]
DS_obs_arr = np.array(cat0['BAT Fluence\n90% Error\n(15-150 keV)\n[10-7 erg/cm2]'])[nonans_mask] / norm.ppf(0.95)
z_arr = np.array(cat0['Redshift'])[nonans_mask]
alpha_arr = np.array(cat0['CPL:alpha'])[nonans_mask]
alpha_arr_u = np.array(cat0['CPL:alpha+'])[nonans_mask]
alpha_arr_d = np.array(cat0['CPL:alpha-'])[nonans_mask]
GRB_names = np.array(cat0['GRB'], dtype=str)[nonans_mask]
GRB_amount = np.sum(nonans_mask)

Dalpha_arr_u = alpha_arr_u - alpha_arr
Dalpha_arr_d = alpha_arr - alpha_arr_d
DE_p_arr_u = E_p_arr_u - E_p_arr
DE_p_arr_d = E_p_arr - E_p_arr_d

#загружаем каталог сверхновых Patheon
catSN = pd.read_csv('catalogues/Pantheon.dat', delimiter='\t', header=0, usecols = [2, 4, 5])

z_arr_SN = np.array(catSN['zcmb'])
mu_arr_SN = np.array(catSN['mu'])# + 19.41623729
Dmu_arr_SN = np.array(catSN['err_mu'])


def loglikelihood(Omega_DE, Omega_k, a, b, k, w):
    
    Omega_m = 1 - Omega_DE + Omega_k

    pars_cosm_planck70 = (70, Omega_m, Omega_DE, Omega_k, w)
    
    #Назначаем выборочные значения для альфы, Ep и Sobs, и сразу же находим значения x и y для плоскости Амати
    print('Создание выборок для гамма-всплесков')
    def loopfun_makesamples(i):
        alpha_sample, E_p_sample, S_obs_sample = make_samples(i, (alpha_arr, Dalpha_arr_d, Dalpha_arr_u, E_p_arr, DE_p_arr_d, DE_p_arr_u, S_obs_arr, DS_obs_arr, z_arr), amount = sample_size)
        Amx_sample, Amy_sample = sample_Amatixy(i, (alpha_sample, E_p_sample, S_obs_sample, z_arr), pars_cosm_planck70)
        return (alpha_sample, E_p_sample, S_obs_sample, Amx_sample, Amy_sample)
        
    alpha_all_samples, E_p_all_samples, S_obs_all_samples, Amx_all_samples, Amy_all_samples = np.array(list(zip(*Parallel(n_jobs=Nthreads, max_nbytes='2048M', verbose=10)(delayed(loopfun_makesamples)(i) for i in np.arange(GRB_amount)))))
    
    #Для полученных выборок находим медианы, а также верхние и нижние пределы
    Amx_meds, Amx_dlim, Amx_ulim = get_meds_and_lims(Amx_all_samples)
    Amy_meds, Amy_dlim, Amy_ulim = get_meds_and_lims(Amy_all_samples)
    
    mu_SN_ufunc = np.vectorize(lambda z: mu(z, pars_cosm_planck70))
    
    Amy_all_samples_corr = Amy_all_samples - np.tensordot(np.log10(z_arr), np.repeat(k, sample_size), axes=0)
    Amy_meds_corr, Amy_dlim_corr, Amy_ulim_corr = get_meds_and_lims(Amy_all_samples_corr)
    Amy_sigma_corr = (Amy_dlim_corr + Amy_ulim_corr) / 2
    
    #Теперь определим mu_A с помощью новых a и b
    print('Расчёт mu_A для выборок гамма-всплесков с коэффициентами a и b, полученными обратной калибровкой')
    def loopfun_sample_mu_A_inv(i):
        return tuple(sample_mu_A(i, (alpha_all_samples[i,:], E_p_all_samples[i,:], S_obs_all_samples[i,:], z_arr), a, b, k))

    mu_A_inv_all_samples = np.array(list(zip(*Parallel(n_jobs=Nthreads, max_nbytes='2048M', verbose=10)(delayed(loopfun_sample_mu_A_inv)(i) for i in np.arange(GRB_amount))))).T
    mu_A_inv_all_samples[~np.isfinite(mu_A_inv_all_samples)] = 0.0
    
    mu_A_inv_meds, mu_A_inv_dlim, mu_A_inv_ulim = get_meds_and_lims(mu_A_inv_all_samples)
    mu_A_inv_sigma = (mu_A_inv_dlim + mu_A_inv_ulim) / 2
    
    return (mu_SN_ufunc(z_arr), mu_A_inv_meds, mu_A_inv_sigma, a * Amx_meds + b, Amy_meds_corr, Amy_sigma_corr)
    
    #посчитаем сумму квадратов невязок в диаграмме Хаббла 
    #chi2_HD = np.sum((mu_SN_ufunc(z_arr) - mu_A_inv_meds) ** 2 / mu_A_inv_sigma ** 2)
    #print(chi2_HD)
    
    #посчитаем сумму квадратов невязок в плоскости Амати
    #chi2_Amati = np.sum((Amy_meds_corr - a * Amx_meds - b) ** 2 / Amy_sigma_corr ** 2)
    #print(chi2_Amati)
    
    #return -0.5 * (chi2_HD + chi2_Amati)

def rho_Am(z):
    C = 0.4
    #return z
    return C ** 2 * 2 * (np.sqrt(1 + z / C ** 2) - 1)
    #return 0.2 * np.log(1 + 5 * z)

def rho_HD(z):
    C = 1.0
    #return z
    return C ** 2 * 2 * (np.sqrt(1 + z / C ** 2) - 1)
    #return 0.5 * np.log(1 + 2 * z)

fixed_realization_alpha = [] #реализация случайных величин исходных наблюдаемых параметров (какая-то реализация для примера)
fixed_realization_E_p = []
fixed_realization_S_obs = []
for i in range(len(z_arr)):
    inp_values = make_samples(i, (alpha_arr, Dalpha_arr_d, Dalpha_arr_u, E_p_arr, DE_p_arr_d, DE_p_arr_u, S_obs_arr, DS_obs_arr, z_arr), amount = 1)
    fixed_realization_alpha.append(inp_values[0][0])
    fixed_realization_E_p.append(inp_values[1][0])
    fixed_realization_S_obs.append(inp_values[2][0])

def loglikelihood_1d(Omega_DE, Omega_k, a, b, k, w, realization_alpha, realization_E_p, realization_S_obs, realization_SN_mu):
    
    #if Omega_DE < 0 or Omega_DE > 1 or Omega_k < -1 or Omega_k > 1: #ограничение на параметры
    #    return -np.inf
    
    Omega_m = 1 - Omega_DE + Omega_k

    pars_cosm_planck70 = (70, Omega_m, Omega_DE, Omega_k, w)
    mu_SN_ufunc = np.vectorize(lambda z: mu(z, pars_cosm_planck70))
    
    Amx_list = []
    Amy_list = []
    mu_A_list = []
    
    for i in range(len(z_arr)): #расчет всех параметров
        z_val = z_arr[i]
        #inp_values = make_samples(i, (alpha_arr, Dalpha_arr_d, Dalpha_arr_u, E_p_arr, DE_p_arr_d, DE_p_arr_u, S_obs_arr, DS_obs_arr, z_arr), amount = 1)
        #alpha_val = inp_values[0][0]
        #E_p_val = inp_values[1][0]
        #S_obs_val = inp_values[2][0]
        alpha_val = realization_alpha[i]
        E_p_val = realization_E_p[i]
        S_obs_val = realization_S_obs[i]
        Amx_val = Amx(z_val, E_p_val)
        Amy_val = Amy(z_val, alpha_val, E_p_val, S_obs_val, pars_cosm_planck70) - np.log10(z_val) * k
        mu_A_val = mu_A(z_val, alpha_val, E_p_val, S_obs_val, a, b, k)
        Amx_list.append(Amx_val)
        Amy_list.append(Amy_val)
        mu_A_list.append(mu_A_val)
    
    Amx_arr = np.array(Amx_list)
    Amy_arr = np.array(Amy_list)
    mu_A_arr = np.array(mu_A_list)
    
    Amx_arr[~np.isfinite(Amx_arr)] = 0
    Amy_arr[~np.isfinite(Amy_arr)] = 0
    mu_A_arr[~np.isfinite(mu_A_arr)] = 0
    
    #посчитаем сумму квадратов невязок в диаграмме Хаббла  для гамма-всплесков
    chi2_HD = np.sum(rho_HD((mu_SN_ufunc(z_arr) - mu_A_arr) ** 2))
    #print(chi2_HD)
    
    #посчитаем сумму квадратов невязок в плоскости Амати
    chi2_Amati = np.sum(rho_Am((Amy_arr - a * Amx_arr - b) ** 2)) * 6.25
    #print(chi2_Amati)
    
    #посчитаем сумму квадратов невязок в диаграмме Хаббла для сверхновых
    chi2_SN = np.sum((mu_SN_ufunc(z_arr_SN) - realization_SN_mu) ** 2)
    
    return -0.5 * (chi2_HD + chi2_Amati + chi2_SN)

def loglikelihood_SNonly(Omega_DE, Omega_k):
    
    Omega_m = 1 - Omega_DE + Omega_k
    
    pars_cosm_planck70 = (73.4, Omega_m, Omega_DE, Omega_k, -1)
    mu_SN_ufunc = np.vectorize(lambda z: mu(z, pars_cosm_planck70))
    
    return -0.5 * np.sum(((mu_SN_ufunc(z_arr_SN) - mu_arr_SN) / Dmu_arr_SN) ** 2)
    
#test = loglikelihood_1d(0.704, 0.0, 0.92, 50.48, 0, -1)
#test2 = loglikelihood_1d(0.704, 0.0, 1, 50, 0, -1)
#test3 = loglikelihood_1d(0.705, 0.0, 1, 50, 0, -1)
#test4 = loglikelihood_1d(0.6, -0.3, 3, 30, 2, -1)

#print(test - test2)
#print(test - test3)
#print(test - test4)

walkers = 300
burnin = 100 
mcmc_iterations = 500

w0 = -3
p0 = (0.7, 0.0, 1, 50, 0)
#sigma_vec = (0.01, 0.01, 0.03, 0.09, 0.06)
sigma_vec = (0.03, 0.03, 0.1, 0.3, 0.2)
p_history = []
for walk in range(walkers):
    
    perwalk_realization_alpha = []
    perwalk_realization_E_p = []
    perwalk_realization_S_obs = []
    for i in range(len(z_arr)): #реализация случайных величин наблюдаемых параметров
        inp_values = make_samples(i, (alpha_arr, Dalpha_arr_d, Dalpha_arr_u, E_p_arr, DE_p_arr_d, DE_p_arr_u, S_obs_arr, DS_obs_arr, z_arr), amount = 1)
        perwalk_realization_alpha.append(inp_values[0][0])
        perwalk_realization_E_p.append(inp_values[1][0])
        perwalk_realization_S_obs.append(inp_values[2][0])
    realization = (perwalk_realization_alpha, perwalk_realization_E_p, perwalk_realization_S_obs, np.random.normal(mu_arr_SN, Dmu_arr_SN))
    
    #Алгоритм Метрополиса-Хастингса
    print(f'Walker number {walk}')
    p =  p0 #начальное приближение параметров
    ll_p = loglikelihood_1d(*p, w0, *realization)
    #ll_p = loglikelihood_SNonly(p, 0)
    for i in tqdm(range(burnin + mcmc_iterations)):
        if i >= burnin:
            p_history.append(p)
        p_candidate = np.random.normal(p, sigma_vec) 
        ll_p_candidate = loglikelihood_1d(*p_candidate, w0, *realization)
        #ll_p_candidate = loglikelihood_SNonly(p_candidate, 0)
        a = np.exp(ll_p_candidate - ll_p)
        if a >= 1:
            p = p_candidate
            ll_p = ll_p_candidate
        else:
            u = np.random.uniform()
            if u <= a:
                p = p_candidate
                ll_p = ll_p_candidate

#fig, ax = plt.subplots()
#ax.plot(np.array(p_history)[:,0], np.array(p_history)[:,1], color='silver', linewidth=0.5)
#ax.scatter(np.array(p_history)[:,0], np.array(p_history)[:,1], s=0.03)
#ax.plot(np.array(p_history)[:,0], np.array(p_history)[:,1])

corner.corner(np.array(p_history), labels = ['Omega_DE', 'Omega_k', 'a', 'b', 'k'])
#corner.corner(np.array(p_history), labels = ['Omega_DE'])

with open('./data/dataGRBw=-3_k.txt', 'w') as f:
    for p in p_history:
        if type(p) == float:
            f.write(str(p)+'\n')
        else:
            f.write(', '.join(p.astype(str))+'\n')

#res_test = curve_fit(np.vectorize(lambda z, Om_DE: mu(z, (70, 1 - Om_DE, Om_DE, 0.0, -1))), z_arr_SN, mu_arr_SN, [0.7], Dmu_arr_SN)

#res_test

walkers = 300
burnin = 100 
mcmc_iterations = 500

p0 = (0.7)
sigma_vec = (0.01)
p_history = []
for walk in range(walkers):
    
#Алгоритм Метрополиса-Хастингса
    print(f'Walker number {walk}')
    p = p0 #начальное приближение параметров
    ll_p = loglikelihood_SNonly(p, 0)
    for i in tqdm(range(burnin + mcmc_iterations)):
        if i >= burnin:
            p_history.append(p)
        p_candidate = np.random.normal(p, sigma_vec) 
        ll_p_candidate = loglikelihood_SNonly(p_candidate, 0)
        a = np.exp(ll_p_candidate - ll_p)
        if a >= 1:
            p = p_candidate
            ll_p = ll_p_candidate
        else:
            u = np.random.uniform()
            if u <= a:
                p = p_candidate
                ll_p = ll_p_candidate

with open('./data/dataSN_new.txt', 'w') as f:
    for p in p_history:
        if type(p) == float:
            f.write(str(p)+'\n')
        else:
            f.write(', '.join(p.astype(str))+'\n')

#fig, ax = plt.subplots()
#ax.plot(np.array(p_history)[:,0], np.array(p_history)[:,1], color='silver', linewidth=0.5)
#ax.scatter(np.array(p_history)[:,0], np.array(p_history)[:,1], s=0.3)
#ax.plot(np.array(p_history)[:,0], np.array(p_history)[:,1])

#corner.corner(np.array(p_history), labels = ['Omega_DE'])


#Снова нарисуем диаграмму Хаббла
#fig, ax = plt.subplots()
#ax.plot(z_log_range, mu_arr_st, c='k', zorder=10)
#ax.errorbar(z_arr, mu_A_inv_meds, yerr=np.array([mu_A_inv_dlim, mu_A_inv_ulim]), linestyle='', linewidth=0.3, marker='o', markersize=1.25, c='teal')
#ax.set_xscale('log')
#ax.set_ylim((32, 52))
#ax.set_title('GRB Hubble Diagram (free $k$)')
#ax.set_xlabel(r'$z$')
#ax.set_ylabel(r'$\mu$')
#fig.tight_layout()
#fig.savefig('pics/HD_icf.png', dpi=300)

#И плоскость Амати
#fig, ax = plt.subplots()
#plot_Amati(fig, ax, (a_inv_sample, b_inv_sample, Amx_flat, Amy_flat_corr, Amx_meds, Amy_meds_corr), 'Amati_icf', 'inv. cosm. fitting (free $k$)', plot_errbars=False, col2 = 'darkblue')
#fig, ax = plt.subplots()
#plot_Amati(fig, ax, (a_inv_sample, b_inv_sample, Amx_meds, Amy_meds_corr, Amx_ulim, Amy_ulim_corr, Amx_dlim, Amy_dlim_corr), 'Amati_icf', 'inv. cosm. fitting (free $k$)', plot_samples=False, col1='teal')
#fig, ax = plt.subplots()
#plot_Amati(fig, ax, (a_inv_sample, b_inv_sample, Amx_flat, Amy_flat_corr, Amx_meds, Amy_meds_corr, Amx_ulim, Amy_ulim_corr, Amx_dlim, Amy_dlim_corr), 'Amati_icf', 'inv. cosm. fitting (free $k$)')

#print('Вычисление хи-квадрат для медианных параметров:')
#print('Для случая варьирования a, b, k: ', str(lcdm_residuals_chi2((a_inv_est, b_inv_est, k_inv_est), z_arr, alpha_arr, E_p_arr, S_obs_arr, Amx_meds, Amy_meds)))

import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner


def log_likelihood(theta, data):
    mu, sigma = theta
    if sigma <= 0:
        return -np.inf
    return -0.5 * np.sum(((data - mu) / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))


def log_prior_wide(theta):
    mu, sigma = theta
    if 0 < mu < 300 and 0 < sigma < 50:
        return 0.0
    return -np.inf


def log_prior_narrow(theta):
    mu, sigma = theta
    if 100 < mu < 110 and 0 < sigma < 50:
        return 0.0
    return -np.inf


def make_log_probability(prior_fn):
    def log_probability(theta, data):
        lp = prior_fn(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, data)

    return log_probability


def run_mcmc(data, initial, prior_fn, label, true_mu, true_sigma):
    print("\n" + "=" * 50)
    print(f"Deney: {label}")
    print("=" * 50)

    n_walkers = 32
    pos = np.array(initial) + 1e-4 * np.random.randn(n_walkers, 2)
    log_prob_fn = make_log_probability(prior_fn)

    sampler = emcee.EnsembleSampler(n_walkers, 2, log_prob_fn, args=(data,))
    sampler.run_mcmc(pos, 2000, progress=True)

    flat_samples = sampler.get_chain(discard=500, thin=15, flat=True)

    mu_q = np.percentile(flat_samples[:, 0], [16, 50, 84])
    sigma_q = np.percentile(flat_samples[:, 1], [16, 50, 84])

    print(f"μ -> median: {mu_q[1]:.2f}, alt: {mu_q[0]:.2f}, üst: {mu_q[2]:.2f}")
    print(f"σ -> median: {sigma_q[1]:.2f}, alt: {sigma_q[0]:.2f}, üst: {sigma_q[2]:.2f}")

    fig = corner.corner(
        flat_samples,
        labels=[r"$\mu$ (Parlaklık)", r"$\sigma$ (Hata)"],
        truths=[true_mu, true_sigma],
    )
    fig.suptitle(label, y=1.02)
    filename = label.lower().replace(" ", "_").replace(":", "") + ".png"
    plt.savefig(filename, bbox_inches="tight")
    plt.show()


def main():
    true_mu = 150.0
    true_sigma = 10.0

    np.random.seed(42)
    data_50 = true_mu + true_sigma * np.random.randn(50)

    run_mcmc(data_50, [140, 5], log_prior_wide,
             "Experiment 1: Initial=140, Wide Prior, n=50", true_mu, true_sigma)

    run_mcmc(data_50, [110, 5], log_prior_wide,
             "Experiment 2: Initial=110, Wide Prior, n=50", true_mu, true_sigma)

    run_mcmc(data_50, [105, 5], log_prior_narrow,
             "Experiment 3: Narrow Prior (mu in 100-110), n=50", true_mu, true_sigma)

    np.random.seed(7)
    data_5 = true_mu + true_sigma * np.random.randn(5)
    run_mcmc(data_5, [140, 5], log_prior_wide,
             "Experiment 4: Wide Prior, n=5", true_mu, true_sigma)


if __name__ == "__main__":
    main()

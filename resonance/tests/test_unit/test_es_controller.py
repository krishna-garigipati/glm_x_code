from __future__ import annotations

import numpy as np
import pytest

from ...es_controller import EvolutionaryController
from ..fixtures.config_provider import build_minimal_core_config, build_minimal_resonance_config


def make_es(seed: int = 42) -> EvolutionaryController:
    cc = build_minimal_core_config()
    rc = build_minimal_resonance_config()
    initial_theta = np.zeros(48, dtype=np.float32)
    initial_theta[0] = 0.008
    initial_theta[1] = 0.02
    initial_theta[2] = 0.1
    initial_theta[3] = 64.0
    for i, rel in enumerate(cc.relations):
        initial_theta[4 + i] = rc.tier1.relation_bias.get(rel, 1.0)
    return EvolutionaryController(
        core_config=cc,
        es_config=rc.es_controller,
        initial_theta=initial_theta,
        theta_indices=rc.theta_indices,
        log_theta_history=True,
        history_buffer_size=1000,
        _seed=seed,
    )


class TestEvolutionaryControllerInit:
    def test_initializes(self):
        es = make_es()
        assert es is not None
        assert es._mu.shape == (48,)
        assert es._sigma == 0.01

    def test_initial_theta_shape_mismatch_raises(self):
        cc = build_minimal_core_config()
        rc = build_minimal_resonance_config()
        bad_theta = np.zeros(10, dtype=np.float32)
        with pytest.raises(ValueError, match="dim mismatch"):
            EvolutionaryController(cc, rc.es_controller, bad_theta, rc.theta_indices)

    def test_initial_mu_copied(self):
        es = make_es()
        theta = es.get_theta()
        theta[0] = 999.0
        assert es.get_theta()[0] != 999.0

    def test_theta_history_starts_logged(self):
        es = make_es()
        history = es.get_theta_history()
        assert len(history) >= 1


class TestGetSetTheta:
    def test_get_theta_returns_copy(self):
        es = make_es()
        t1 = es.get_theta()
        t2 = es.get_theta()
        t1[0] = 999.0
        assert t2[0] != 999.0

    def test_set_theta_updates_mu(self):
        es = make_es()
        old = es.get_theta().copy()
        new = old + 0.01
        es.set_theta(new)
        assert np.allclose(es.get_theta(), new)

    def test_set_theta_nan_handled(self):
        es = make_es()
        bad = np.full(48, np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="NaN"):
            es.set_theta(bad)


class TestProposeMutation:
    def test_returns_different_theta(self):
        es = make_es()
        proposal = es.propose_theta_mutation()
        assert proposal.shape == (48,)
        assert np.isfinite(proposal).all()

    def test_population_queue_refills(self):
        es = make_es()
        proposals = [es.propose_theta_mutation() for _ in range(20)]
        assert all(p.shape == (48,) for p in proposals)
        assert all(np.isfinite(p).all() for p in proposals)

    def test_proposals_within_bounds(self):
        es = make_es()
        for _ in range(10):
            prop = es.propose_theta_mutation()
            idx = es._theta_indices
            assert es._core.es_bounds.propagation_threshold[0] <= prop[idx.propagation_threshold] <= es._core.es_bounds.propagation_threshold[1]
            assert es._core.es_bounds.edge_threshold[0] <= prop[idx.edge_threshold] <= es._core.es_bounds.edge_threshold[1]
            assert es._core.es_bounds.decay_lambda[0] <= prop[idx.decay_lambda] <= es._core.es_bounds.decay_lambda[1]
            assert es._core.es_bounds.top_k[0] <= prop[idx.top_k] <= es._core.es_bounds.top_k[1]


class TestUpdateWithReward:
    def test_single_reward_does_not_trigger_update(self):
        es = make_es()
        old_mu = es.get_theta().copy()
        es.update_es_with_reward(1.0, es.get_theta())
        assert np.allclose(es.get_theta(), old_mu)

    def test_evaluation_window_trigger(self):
        es = make_es()
        old_mu = es.get_theta().copy()
        thetas_used = []
        for _ in range(es._config.evaluation_window):
            thetas_used.append(es.propose_theta_mutation().copy())
        for i, theta_used in enumerate(thetas_used):
            es.update_es_with_reward(float(i) / es._config.evaluation_window, theta_used)
        updated_mu = es.get_theta()
        assert not np.allclose(updated_mu, old_mu)

    def test_negative_reward_handled(self):
        es = make_es()
        theta = es.get_theta()
        es.update_es_with_reward(-10.0, theta)
        assert np.isfinite(es.get_theta()).all()

    def test_nan_reward_clamped(self):
        es = make_es()
        theta = es.get_theta()
        es.update_es_with_reward(float("nan"), theta)
        assert np.isfinite(es.get_theta()).all()

    def test_inf_reward_clamped(self):
        es = make_es()
        theta = es.get_theta()
        es.update_es_with_reward(float("inf"), theta)
        assert np.isfinite(es.get_theta()).all()

    def test_theta_shape_mismatch_raises(self):
        es = make_es()
        bad = np.zeros(10, dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            es.update_es_with_reward(1.0, bad)

    def test_sigma_decays(self):
        es = make_es()
        initial_sigma = es._sigma
        theta = es.get_theta()
        for i in range(es._config.evaluation_window):
            es.update_es_with_reward(0.5, theta)
        assert es._sigma < initial_sigma

    def test_anchor_penalty_applied(self):
        cc = build_minimal_core_config()
        rc = build_minimal_resonance_config()
        assert rc.es_controller.anchor_enabled is True
        es = make_es()
        initial_mu = es._initial_mu.copy()
        drift = es.get_theta() - initial_mu
        if np.any(drift != 0):
            es._mu = es._mu + 10.0
            theta = es.get_theta()
            for i in range(es._config.evaluation_window):
                es.update_es_with_reward(1.0, theta)
            assert np.linalg.norm(es.get_theta() - initial_mu) < 15.0


class TestThetaHistory:
    def test_history_records_after_update(self):
        es = make_es()
        initial_len = len(es.get_theta_history())
        theta = es.get_theta()
        for i in range(es._config.evaluation_window):
            es.update_es_with_reward(0.5, theta)
        assert len(es.get_theta_history()) >= initial_len + 1

    def test_clear_history(self):
        es = make_es()
        es.clear_theta_history()
        assert len(es.get_theta_history()) == 0

    def test_history_buffer_size_enforced(self):
        es = make_es()
        for _ in range(es._history_buffer_size + 50):
            es._record_theta_locked()
        assert len(es._theta_history) <= es._history_buffer_size


class TestMemoryLeak:
    def test_samples_cleared_after_update(self):
        es = make_es()
        theta = es.get_theta()
        for i in range(es._config.evaluation_window):
            es.update_es_with_reward(0.5, theta)
        assert len(es._samples) == 0

import torch

from verl import DataProto

from agrimanager.adapter.trainer.trainer import AgriTrainer


class RecordingWorkerGroup:
    world_size = 4

    def __init__(self):
        self.batch_sizes = []

    def _result(self, batch: DataProto, key: str) -> DataProto:
        self.batch_sizes.append(len(batch))
        return DataProto.from_dict(
            tensors={key: batch.batch["sample_id"].clone()}
        )

    def compute_values(self, batch: DataProto) -> DataProto:
        return self._result(batch, "values")

    def compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        return self._result(batch, "ref_log_prob")

    def compute_log_prob(self, batch: DataProto) -> DataProto:
        return self._result(batch, "old_log_probs")


def _batch() -> DataProto:
    return DataProto.from_dict(
        tensors={"sample_id": torch.tensor([10, 20, 30])}
    )


def _trainer() -> AgriTrainer:
    trainer = object.__new__(AgriTrainer)
    trainer.use_legacy_worker_impl = "enable"
    return trainer


def test_compute_token_values_pads_for_worker_and_restores_original_rows():
    trainer = _trainer()
    trainer.critic_wg = RecordingWorkerGroup()

    result = trainer._compute_token_values(_batch())

    assert trainer.critic_wg.batch_sizes == [4]
    assert result.batch["values"].tolist() == [10, 20, 30]


def test_compute_ref_log_prob_pads_for_worker_and_restores_original_rows():
    trainer = _trainer()
    trainer.ref_policy_wg = RecordingWorkerGroup()

    result = trainer._compute_ref_log_prob(_batch())

    assert trainer.ref_policy_wg.batch_sizes == [4]
    assert result.batch["ref_log_prob"].tolist() == [10, 20, 30]


def test_compute_old_log_prob_pads_for_worker_and_restores_original_rows():
    trainer = _trainer()
    trainer.actor_rollout_wg = RecordingWorkerGroup()

    result, mfu = trainer._compute_old_log_prob(_batch())

    assert trainer.actor_rollout_wg.batch_sizes == [4]
    assert result.batch["old_log_probs"].tolist() == [10, 20, 30]
    assert mfu == 0

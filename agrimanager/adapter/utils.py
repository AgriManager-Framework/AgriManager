"""Utilities for AgriManager adapter.

Provides AgriGenerationsLogger for logging validation episode trajectories
to wandb tables with a row-per-episode format.
"""

import dataclasses


@dataclasses.dataclass
class AgriGenerationsLogger:
    """Logs validation episode data to a wandb table.

    Unlike verl's ValidationGenerationsLogger (column-per-sample), this uses
    a row-per-episode format for easier filtering and sorting in wandb.

    Table columns include validation labels so W&B tables can be filtered by
    the experiment's OOD axis and by individual scenario.
    """

    _columns: list = dataclasses.field(
        default_factory=lambda: [
            "step",
            "validation_set",
            "validation_axis",
            "validation_axis_value",
            "weather_regime",
            "crop_regime",
            "scenario_id",
            "crop_name",
            "trajectory",
            "reward",
            "num_steps",
        ],
        init=False,
        repr=False,
    )

    def log(self, loggers, samples, step):
        """Log episode samples to configured loggers.

        Args:
            loggers: List of logger names (e.g. ["console", "wandb"]).
            samples: List of dicts with keys: trajectory, reward, num_steps.
            step: Current training step.
        """
        if "wandb" in loggers:
            self._log_to_wandb(samples, step)

    def _log_to_wandb(self, samples, step):
        """Log episode data to a cumulative wandb.Table at 'val/agri_generations'.

        Uses the table recreation workaround for wandb issue #2981.
        """
        import wandb

        if not hasattr(self, "_wandb_table"):
            self._wandb_table = wandb.Table(columns=self._columns)

        # Recreate table with existing data (wandb issue #2981 workaround)
        new_table = wandb.Table(columns=self._columns, data=self._wandb_table.data)

        for sample in samples:
            new_table.add_data(
                step,
                sample.get("validation_set", ""),
                sample.get("validation_axis", ""),
                sample.get("validation_axis_value", ""),
                sample.get("weather_regime", ""),
                sample.get("crop_regime", ""),
                sample.get("scenario_id", ""),
                sample.get("crop_name", ""),
                sample.get("trajectory", ""),
                sample.get("reward", 0.0),
                sample.get("num_steps", 0),
            )

        if wandb.run is not None:
            wandb.log({"val-table/agri_generations": new_table}, step=step)
        self._wandb_table = new_table

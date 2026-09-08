from dataclasses import dataclass
from typing import Literal, Optional, TypeAlias, cast

from core.metrics.enums import ReduceProtocol

Path: TypeAlias = tuple[str | ReduceProtocol, ...]


@dataclass(frozen=True, slots=True)
class Query:
    """Defines a reporting query over a MetricSchema."""

    title: str
    x: Path
    y: Path | tuple[Path, ...]
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    legend_labels: Optional[tuple[str, ...]] = None
    error: Literal["none", "std"] = "none"
    error_path: Optional[Path] = None

    def __post_init__(self) -> None:
        if not self.x:
            raise ValueError("Query x path cannot be empty.")

        if not self.y:
            raise ValueError("Query y path cannot be empty.")

        if self.legend_labels is not None and len(self.legend_labels) != len(
            self.y_paths
        ):
            raise ValueError("legend_labels must have the same length as y paths.")

        if self.error not in ("none", "std"):
            raise ValueError(f"Unsupported error statistic: {self.error!r}.")

        if self.error == "none":
            if self.error_path is not None:
                raise ValueError("error_path requires an error statistic.")

            return

        if not self.error_path:
            raise ValueError(f"error={self.error!r} requires error_path.")

        if isinstance(self.error_path[-1], ReduceProtocol):
            raise ValueError(
                "error_path must point to the dynamic dimension, not its reduction operator."
            )

        target = self.error_path + (ReduceProtocol.MEAN,)

        if not any(path[: len(target)] == target for path in self.y_paths):
            raise ValueError(
                f"error_path {self.error_path} must be followed by ReduceProtocol.MEAN in one of "
                "the query's y paths."
            )

    @property
    def y_paths(
        self,
    ) -> tuple[Path, ...]:
        if self.y and isinstance(self.y[0], tuple):
            return cast(tuple[Path, ...], self.y)

        return (cast(Path, self.y),)

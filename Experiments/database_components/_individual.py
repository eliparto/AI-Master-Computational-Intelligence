"""Individual class."""

from dataclasses import dataclass
from typing import List

import sqlalchemy.orm as orm
import sqlalchemy
from sqlalchemy.types import JSON
from sqlalchemy.ext.mutable import MutableList


from revolve2.experimentation.optimization.ea import Individual as GenericIndividual

from ._base import Base
from ._genotype import Genotype


@dataclass
class Individual(
    Base, GenericIndividual[Genotype], population_table="population", kw_only=True
):
    """An individual in a population."""

    __tablename__ = "individual"
    old_fitness: orm.Mapped[float] = orm.mapped_column(nullable=False, default=0.0, init=False)
    fitness_history: orm.Mapped[List[List[float]]] = orm.mapped_column(
        MutableList.as_mutable(JSON), default=list, nullable=False
    )




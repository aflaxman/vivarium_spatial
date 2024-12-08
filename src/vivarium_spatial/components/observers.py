from typing import List
import pandas as pd
from pandas.api.types import CategoricalDtype
from vivarium.framework.engine import Builder
from vivarium_public_health.results.observer import PublicHealthObserver
from vivarium_public_health.results.columns import COLUMNS


class CollisionObserver(PublicHealthObserver):
    """Observes particle collisions in the simulation."""
    
    ##############
    # Properties #
    ##############

    @property 
    def configuration_defaults(self):
        return {
            "stratification": {
                "collisions": super().configuration_defaults["stratification"][
                    self.get_configuration_name()
                ]
            }
        }

    @property
    def columns_required(self) -> List[str]:
        return ["collision_count", "last_collision_time", "whimsy"]

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.clock = builder.time.clock()
        self.register_stratifications(builder)

    #################
    # Setup methods #
    #################

    def get_configuration(self, builder: Builder):
        return builder.configuration.stratification.collisions

    def register_stratifications(self, builder: Builder) -> None:
        """Register stratifications used by this observer"""
        builder.results.register_stratification(
            name='whimsy',
            categories=['low', 'medium', 'high'],
            mapper=self.map_whimsy_categories,
            requires_columns=['whimsy'],
            is_vectorized=True
        )

    def register_observations(self, builder: Builder):
        self.register_adding_observation(
            builder=builder,
            name="collision_counts",
            pop_filter='',
            when="collect_metrics",
            requires_columns=["collision_count", "last_collision_time", "whimsy"],
            additional_stratifications=self.configuration.include,
            excluded_stratifications=self.configuration.exclude,
            aggregator=self.aggregate_collisions
        )

    def map_whimsy_categories(self, pop: pd.DataFrame) -> pd.Series:
        """Maps whimsy values to categories using fixed cut points"""
        bins = [0, 1/3, 2/3, 1]
        return pd.cut(pop.whimsy, bins=bins, labels=['low', 'medium', 'high'], include_lowest=True)

    ###############
    # Aggregators #
    ###############

    def aggregate_collisions(self, pop: pd.DataFrame) -> float:
        """Aggregates collision counts for particles that collided in current timestep"""
        current_time = self.clock()
        collided_this_step = pop.last_collision_time == current_time
        return len(pop[collided_this_step])

    ##############################
    # Results formatting methods #
    ##############################

    def format(self, measure: str, results: pd.DataFrame) -> pd.DataFrame:
        """Format the results dataframe"""
        results = results.reset_index()
        if 'whimsy' in results.columns:
            results.rename(columns={'whimsy': COLUMNS.SUB_ENTITY}, inplace=True)
        return results

    def get_measure_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        """Get the measure column values"""
        return pd.Series('collision_count', index=results.index)

    def get_entity_type_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        """Get the entity_type column values"""
        return pd.Series('particle', index=results.index)

    def get_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        """Get the entity column values"""
        return pd.Series('collision', index=results.index)

    def get_sub_entity_column(self, measure: str, results: pd.DataFrame) -> pd.Series:
        """Get the sub_entity column values"""
        if COLUMNS.SUB_ENTITY in results.columns:
            return results[COLUMNS.SUB_ENTITY].astype(CategoricalDtype())
        return pd.Series('all', index=results.index, dtype=CategoricalDtype())
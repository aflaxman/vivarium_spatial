from typing import List
import pandas as pd
from vivarium.framework.engine import Builder
from vivarium_public_health.results.observer import PublicHealthObserver


class CollisionObserver(PublicHealthObserver):
    """Observes particle collisions in the simulation.
    
    This observer computes collision counts for each timestep by tracking
    particles that collided during the current timestep only.
    """
    
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
        return ["collision_count", "last_collision_time"]

    #####################
    # Lifecycle methods #
    #####################

    def setup(self, builder: Builder) -> None:
        self.clock = builder.time.clock()

    #################
    # Setup methods #
    #################

    def get_configuration(self, builder: Builder):
        return builder.configuration.stratification.collisions

    def register_observations(self, builder: Builder):
        self.register_adding_observation(
            builder=builder,
            name="collision_counts",
            pop_filter='',
            when="collect_metrics",
            requires_columns=["collision_count", "last_collision_time"],
            additional_stratifications=self.configuration.include,
            excluded_stratifications=self.configuration.exclude,
            aggregator=self.aggregate_collisions
        )

    ###############
    # Aggregators #
    ###############

    def aggregate_collisions(self, pop: pd.DataFrame) -> float:
        """Aggregates collision counts for particles that collided in current timestep"""
        current_time = self.clock()
        collided_this_step = pop.last_collision_time == current_time
        return len(pop[collided_this_step])
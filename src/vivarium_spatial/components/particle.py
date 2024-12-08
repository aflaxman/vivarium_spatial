from typing import List
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
from vivarium.framework.event import Event

class Basic(Component):
    name = 'basic'

    @property
    def columns_created(self):
        return ['x', 'y', 'theta']
    
    def setup(self, builder):
        self.near_radius = 1
        self.max_step_size = 0.01
        
        # Register the max angle change pipeline
        self.max_angle_change = builder.value.register_value_producer(
            'particle.max_angle_change',
            source=lambda index: pd.Series(10.0, index=index)
        )

        self.randomness = builder.randomness.get_stream('particle.basic')
                
    def on_initialize_simulants(self, simulant_data):
        """Start new simulants at random location in unit square,
        with random direction in degrees [0, 360)
        
        Parameters
        ----------
        simulant_data : SimulantData
            Data about the new simulants to initialize
        """
        pop = pd.DataFrame(index=simulant_data.index)
        
        # Generate random x,y coordinates in [0,1) x [0,1)
        pop['x'] = self.randomness.get_draw(pop.index, additional_key='x')
        pop['y'] = self.randomness.get_draw(pop.index, additional_key='y')
        
        # Generate random angles in [0, 360) degrees
        pop['theta'] = self.randomness.get_draw(pop.index, additional_key='theta') * 360.0
        
        self.population_view.update(pop)

    def on_time_step(self, event):
        """Update particle positions by moving random amounts in their theta directions"""
        pop = self.population_view.get(event.index)
        
        # Generate random step sizes between 0 and max_step_size
        steps = self.randomness.get_draw(pop.index, additional_key='step') * self.max_step_size
        
        # Convert angles to radians for calculation
        theta_rad = np.deg2rad(pop['theta'])
        
        # Calculate position updates using trigonometry
        dx = steps * np.cos(theta_rad)
        dy = steps * np.sin(theta_rad)
        
        # Get max angle change from pipeline, which may be modified by whimsy
        max_angle = self.max_angle_change(pop.index)
        dtheta = (self.randomness.get_draw(pop.index, additional_key='angle')-.5) * 2 * max_angle
        
        # Update positions
        pop['x'] = (pop['x'] + dx) % 1.0  # Wrap around unit square using modulo
        pop['y'] = (pop['y'] + dy) % 1.0
        pop['theta'] = (pop['theta'] + dtheta) % 360.0
        
        self.population_view.update(pop)
        
        return pop

class Whimsy(Component):
    """Component that gives each particle a whimsy value and modifies their max angle change"""
    
    @property
    def columns_created(self):
        return ['whimsy']
    
    def setup(self, builder):
        self.randomness = builder.randomness.get_stream('particle.whimsy')
        
        # Register modifier for max angle change pipeline
        builder.value.register_value_modifier(
            'particle.max_angle_change',
            modifier=self.modify_max_angle_change,
            requires_columns=['whimsy']
        )
    
    def on_initialize_simulants(self, simulant_data):
        """Initialize whimsy values uniformly between 0 and 1"""
        pop = pd.DataFrame({
            'whimsy': self.randomness.get_draw(simulant_data.index)
        })
        self.population_view.update(pop)
    
    def modify_max_angle_change(self, index, angle):
        """Modify max angle change based on whimsy.
        Low whimsy particles will have very small angle changes,
        high whimsy particles will have larger angle changes."""
        pop = self.population_view.get(index)
        # Scale the base angle change by whimsy value
        return angle * pop.whimsy
class Collisions(Component):
    """Component for handling particle collisions using KDTree for efficient neighbor finding"""
    
    @property
    def columns_created(self) -> List[str]:
        return ['collision_count', 'last_collision_time']
        
    @property 
    def columns_required(self) -> List[str]:
        return ['x', 'y', 'theta']

    def __init__(self):
        super().__init__()
        self.current_collisions = []  # List to store current collision locations
        
    def setup(self, builder: Builder) -> None:
        """Setup collision parameters and randomness stream"""
        self.critical_radius = 0.01  # Distance threshold for collision detection
        self.randomness = builder.randomness.get_stream('particle.collisions')
        self.clock = builder.time.clock()
        
    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        self.population_view.update(pd.DataFrame({
            'collision_count': 0,
            'last_collision_time': pd.NaT
        }, index=pop_data.index))
        
    def on_time_step(self, event: Event) -> None:
        """Check for collisions and update particle directions"""
        pop = self.population_view.get(event.index)
        if len(pop) < 2:  # Need at least 2 particles for collisions
            return
            
        # Build KDTree from current particle positions
        positions = pop[['x', 'y']].values
        tree = KDTree(positions, leaf_size=2)
        
        # Find all pairs of particles within critical radius
        collision_pairs = tree.query_radius(positions, r=self.critical_radius, count_only=True)
        
        # Get particles with neighbors (arrays with length > 1)
        collided_particles, = np.where(collision_pairs > 1)
        
        if len(collided_particles) == 0:
            self.current_collisions = []  # Clear current collisions
            return

        # Store collision locations for visualization
        self.current_collisions = [(pop.iloc[i]['x'], pop.iloc[i]['y']) 
                                 for i in collided_particles]

        # Generate new random angles for all collided particles
        new_angles = self.randomness.get_draw(
            pd.Index(collided_particles),
            additional_key='collision_theta'
        ) * 360.0            
        
        # Update collision stats
        updates = pd.DataFrame({
            'theta': new_angles,
            'collision_count': pop.loc[new_angles.index, 'collision_count'] + 1,
            'last_collision_time': pd.Series(self.clock(), index=new_angles.index)
        })
        
        self.population_view.update(updates)
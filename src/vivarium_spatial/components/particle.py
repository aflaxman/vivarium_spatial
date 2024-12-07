import numpy as np
import pandas as pd
import sklearn.neighbors
import networkx as nx
from vivarium import Component

class Basic(Component):
    name = 'basic'

    @property
    def columns_created(self):
        return ['x', 'y', 'theta']
    
    def setup(self, builder):
        self.near_radius = 1
        self.max_step_size = 0.1

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
        
        # Update positions
        pop['x'] = (pop['x'] + dx) % 1.0  # Wrap around unit square using modulo
        pop['y'] = (pop['y'] + dy) % 1.0
        
        # # Get current positions for neighbor calculation
        # positions = pop[['x', 'y']].values
        
        # # Find neighbors within radius
        # nbrs = sklearn.neighbors.NearestNeighbors(radius=self.near_radius)
        # nbrs.fit(positions)
        # neighbor_graph = nbrs.radius_neighbors_graph(positions)
        
        # # Convert to networkx graph for analysis
        # G = nx.from_scipy_sparse_array(neighbor_graph)
        
        # Update the population state
        self.population_view.update(pop)
        
        return pop

    def near_frozen(self, pop):
        not_frozen = pop[pop.frozen.isnull()].loc[:, ['x', 'y', 'z']]
        if len(not_frozen) == 0:
            return pd.Series(dtype='float64')

        frozen = pop[~pop.frozen.isnull()].loc[:, ['x', 'y', 'z']]
        X = frozen.values
        
        tree = sklearn.neighbors.KDTree(X, leaf_size=2)
        
        num_near = tree.query_radius(not_frozen.values, r=self.config.near_radius, count_only=True)
        to_freeze = not_frozen[(num_near > 0)].index
        if len(to_freeze) == 0:
            return pd.Series(dtype='float64')
        index_near = tree.query_radius(not_frozen.loc[to_freeze].values, r=self.config.near_radius, count_only=False)
        
        return pd.Series(map(lambda x:frozen.index[x[0]], # HACK: get the index of the first frozen node close to this one
                             index_near), index=to_freeze)

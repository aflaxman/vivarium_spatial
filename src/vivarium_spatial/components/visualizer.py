import pygame
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

class ParticleVisualizer(Component):
    """A component that visualizes particles in a unit square using Pygame."""
    
    @property
    def columns_required(self) -> Optional[List[str]]:
        return ['x', 'y', 'theta']

    def __init__(self, 
                 background_color: tuple = (0, 0, 0),
                 particle_color: tuple = (155, 155, 255),
                 progress_color: tuple = (0, 255, 0),
                 border_color: tuple = (100, 100, 100),
                 border_width: int = 2):
        """
        Parameters
        ----------
        background_color : tuple
            RGB color for the background
        particle_color : tuple
            RGB color for particles
        progress_color : tuple
            RGB color for the progress bar
        border_color : tuple
            RGB color for the unit square border
        border_width : int
            Width of the border in pixels
        """
        super().__init__()
        self.background_color = background_color
        self.particle_color = particle_color
        self.progress_color = progress_color
        self.border_color = border_color
        self.border_width = border_width
        self._screen = None
        
        # Will be set in setup
        self.scale = None
        self.view_offset = None
        self.simulation_bounds = None
        
    def setup(self, builder: Builder) -> None:
        """Initialize pygame and set up the display window."""
        pygame.init()
        
        # Get the current screen info for fullscreen mode
        screen_info = pygame.display.Info()
        self.width = screen_info.current_w
        self.height = screen_info.current_h
        self._screen = pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN)
        pygame.display.set_caption("Particle Simulation")
        
        # Store simulation time information for progress bar
        self.start_time = pd.Timestamp(**builder.configuration.time.start)
        self.end_time = pd.Timestamp(**builder.configuration.time.end)
        self.clock = builder.time.clock()
        
        # Calculate view parameters for unit square
        self._setup_unit_square_view()

    def _setup_unit_square_view(self) -> None:
        """Set up the viewing parameters for the unit square."""
        # Parameters
        padding = 50
        top_margin = 10  # space for progress bar
        
        # Calculate available display area
        display_width = self.width - 2 * padding
        display_height = self.height - 2 * padding - top_margin
        
        # Calculate scale to fit unit square while maintaining aspect ratio
        self.scale = min(display_width, display_height)
        
        # Calculate offsets to center the unit square
        x_offset = (self.width - self.scale) / 2
        y_offset = (self.height - self.scale) / 2 + top_margin
        self.view_offset = (x_offset, y_offset)
        
        # Store the screen coordinates of the unit square corners for border drawing
        self.simulation_bounds = pygame.Rect(
            x_offset, y_offset, self.scale, self.scale
        )

    def sim_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert simulation coordinates (in unit square) to screen coordinates."""
        screen_x = self.view_offset[0] + x * self.scale
        screen_y = self.view_offset[1] + y * self.scale
        return int(screen_x), int(screen_y)

    def on_time_step(self, event: Event) -> None:
        """Update visualization each time step."""
        if self.scale is None:
            return
            
        pop = self.population_view.get(event.index)
        if pop.empty:
            return
        
        self._screen.fill(self.background_color)
        
        # Draw progress bar first
        self._draw_progress_bar()
        
        # Draw unit square border
        self._draw_border()
        
        # Then draw particles
        self._draw_particles(pop)
        
        pygame.display.flip()
        
        # Handle window events
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key in [pygame.K_ESCAPE, pygame.K_q]
            ):
                pygame.quit()
                
    def _draw_particles(self, population: pd.DataFrame) -> None:
        """Draw all particles with their current positions and directions."""
        for _, particle in population.iterrows():
            # Convert to screen coordinates
            px, py = self.sim_to_screen(particle['x'], particle['y'])
            
            # Draw particle
            pygame.draw.circle(self._screen, self.particle_color, (px, py), 3)
            
            # Draw direction indicator
            angle_rad = np.deg2rad(particle['theta'])
            line_length = 10
            end_x = px + line_length * np.cos(angle_rad)
            end_y = py + line_length * np.sin(angle_rad)
            pygame.draw.line(self._screen, self.particle_color,
                           (px, py), 
                           (int(end_x), int(end_y)), 1)
            
    def _draw_border(self) -> None:
        """Draw the border of the unit square."""
        pygame.draw.rect(self._screen, self.border_color, 
                        self.simulation_bounds, self.border_width)
            
    def _draw_progress_bar(self) -> None:
        """Draw a progress bar at the top of the screen."""
        # Calculate progress as fraction of simulation time elapsed
        current_time = self.clock()
        progress = (current_time - self.start_time) / (self.end_time - self.start_time)
        
        # Draw progress bar
        bar_height = 3
        bar_width = int(self.width * progress)
        progress_rect = pygame.Rect(0, 0, bar_width, bar_height)
        pygame.draw.rect(self._screen, self.progress_color, progress_rect)
import pygame
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

class ParticleVisualizer(Component):
    """A component that visualizes particles with motion trails in a unit square."""
    
    @property
    def columns_required(self) -> Optional[List[str]]:
        return ['x', 'y', 'theta', 'whimsy']

    def __init__(self, 
                 background_color: tuple = (0, 0, 0),
                 progress_color: tuple = (0, 255, 0),
                 border_color: tuple = (100, 100, 100),
                 border_width: int = 2,
                 fade_speed: float = 0.005,
                 triangle_size: int = 6):
        """
        Parameters
        ----------
        background_color : tuple
            RGB color for the background
        progress_color : tuple
            RGB color for the progress bar
        border_color : tuple
            RGB color for the unit square border
        border_width : int
            Width of the border in pixels
        fade_speed : float
            Rate at which trails fade (0-1)
        triangle_size : int
            Base size of the triangle particles
        """
        super().__init__()
        self.background_color = background_color
        self.progress_color = progress_color
        self.border_color = border_color
        self.border_width = border_width
        self.fade_speed = fade_speed
        self.triangle_size = triangle_size
        
        # Define color gradient endpoints (tasteful blue to red)
        self.low_whimsy_color = (41, 98, 255)    # Cool blue
        self.high_whimsy_color = (255, 89, 94)   # Warm red
        
        self._screen = None
        self._trail_surface = None
        
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
        
        # Create trail surface with alpha channel
        self._trail_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self._trail_surface.fill((*self.background_color, 255))  # Include alpha channel
        
        # Store simulation time information for progress bar
        self.start_time = pd.Timestamp(**builder.configuration.time.start)
        self.end_time = pd.Timestamp(**builder.configuration.time.end)
        self.clock = builder.time.clock()
        
        # Calculate view parameters for unit square
        self._setup_unit_square_view()

    def _setup_unit_square_view(self) -> None:
        """Set up the viewing parameters for the unit square."""
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

    def get_color_for_whimsy(self, whimsy: float) -> Tuple[int, int, int]:
        """Calculate color based on whimsy value using linear interpolation."""
        return tuple(int(self.low_whimsy_color[i] + (self.high_whimsy_color[i] - self.low_whimsy_color[i]) * whimsy)
                    for i in range(3))
    
    def calculate_triangle_points(self, x: float, y: float, theta: float) -> List[Tuple[int, int]]:
        """Calculate the three points of an isosceles triangle facing in the direction of theta."""
        # Convert angle to radians
        angle_rad = np.deg2rad(theta)
        
        # Calculate the three points of the isosceles triangle
        # The base is perpendicular to the direction, and the point faces the direction
        
        # Forward point
        front_x = x + self.triangle_size * 1.5 * np.cos(angle_rad) / self.scale
        front_y = y + self.triangle_size * 1.5 * np.sin(angle_rad) / self.scale
        
        # Back points (perpendicular to direction)
        back_angle_1 = angle_rad + np.pi/2
        back_angle_2 = angle_rad - np.pi/2
        
        back1_x = x + self.triangle_size * np.cos(back_angle_1) / self.scale
        back1_y = y + self.triangle_size * np.sin(back_angle_1) / self.scale
        
        back2_x = x + self.triangle_size * np.cos(back_angle_2) / self.scale
        back2_y = y + self.triangle_size * np.sin(back_angle_2) / self.scale
        
        # Convert all points to screen coordinates
        screen_points = [
            self.sim_to_screen(front_x, front_y),
            self.sim_to_screen(back1_x, back1_y),
            self.sim_to_screen(back2_x, back2_y)
        ]
        
        return screen_points

    def on_time_step(self, event: Event) -> None:
        """Update visualization each time step."""
        if self.scale is None:
            return
            
        pop = self.population_view.get(event.index)
        if pop.empty:
            return
        
        # Create fade surface
        fade_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        fade_surface.fill((0, 0, 0, int(255 * self.fade_speed)))
        
        # Apply fade effect
        self._trail_surface.blit(fade_surface, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)
        
        # Draw current frame to trail surface
        self._draw_frame_to_trail(pop)
        
        # Draw everything to screen
        self._screen.fill(self.background_color)
        self._screen.blit(self._trail_surface, (0, 0))
        self._draw_border()
        self._draw_progress_bar()
        
        pygame.display.flip()
        
        # Handle window events
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key in [pygame.K_ESCAPE, pygame.K_q]
            ):
                pygame.quit()
                
    def _draw_frame_to_trail(self, population: pd.DataFrame) -> None:
        """Draw the current frame to the trail surface."""
        for _, particle in population.iterrows():
            # Get color based on whimsy
            particle_color = self.get_color_for_whimsy(particle['whimsy'])
            
            # Calculate triangle points
            triangle_points = self.calculate_triangle_points(
                particle['x'], 
                particle['y'], 
                particle['theta']
            )
            
            # Draw filled triangle
            pygame.draw.polygon(self._trail_surface, particle_color, triangle_points)
            
    def _draw_border(self) -> None:
        """Draw the border of the unit square."""
        pygame.draw.rect(self._screen, self.border_color, 
                        self.simulation_bounds, self.border_width)
            
    def _draw_progress_bar(self) -> None:
        """Draw a progress bar at the top of the screen."""
        current_time = self.clock()
        progress = (current_time - self.start_time) / (self.end_time - self.start_time)
        
        bar_height = 3
        bar_width = int(self.width * progress)
        progress_rect = pygame.Rect(0, 0, bar_width, bar_height)
        pygame.draw.rect(self._screen, self.progress_color, progress_rect)
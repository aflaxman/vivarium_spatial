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
                 fade_speed: float = 0.02,
                 triangle_size: int = 6,
                 trail_size: int = 5,
                 trail_length: int = 5):  # How many previous positions to keep
        super().__init__()
        self.background_color = background_color
        self.progress_color = progress_color
        self.border_color = border_color
        self.border_width = border_width
        self.fade_speed = fade_speed
        self.triangle_size = triangle_size
        self.trail_size = trail_size
        self.trail_length = trail_length
        
        # Dictionary to store previous positions
        self.particle_history = {}
        
        # Define color gradient endpoints
        self.low_whimsy_color = (41, 98, 255)    # Cool blue
        self.high_whimsy_color = (255, 89, 94)   # Warm red
        
        self._screen = None
        self._trail_surface = None
        self._current_surface = None
        
        self.scale = None
        self.view_offset = None
        self.simulation_bounds = None
        
    def setup(self, builder: Builder) -> None:
        pygame.init()
        
        screen_info = pygame.display.Info()
        self.width = screen_info.current_w
        self.height = screen_info.current_h
        self._screen = pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN)
        pygame.display.set_caption("Particle Simulation")
        
        # Create trail surface with alpha channel for fading trails
        self._trail_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self._trail_surface.fill((0, 0, 0, 0))  # Fully transparent initially
        
        # Create surface for current particle positions
        self._current_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        self.start_time = pd.Timestamp(**builder.configuration.time.start)
        self.end_time = pd.Timestamp(**builder.configuration.time.end)
        self.clock = builder.time.clock()
        
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
        angle_rad = np.deg2rad(theta)
        
        # Forward point
        front_x = x + self.triangle_size * 1.5 * np.cos(angle_rad) / self.scale
        front_y = y + self.triangle_size * 1.5 * np.sin(angle_rad) / self.scale
        
        # Back points
        back_angle_1 = angle_rad + np.pi/2
        back_angle_2 = angle_rad - np.pi/2
        
        back1_x = x + self.triangle_size * np.cos(back_angle_1) / self.scale
        back1_y = y + self.triangle_size * np.sin(back_angle_1) / self.scale
        
        back2_x = x + self.triangle_size * np.cos(back_angle_2) / self.scale
        back2_y = y + self.triangle_size * np.sin(back_angle_2) / self.scale
        
        return [
            self.sim_to_screen(front_x, front_y),
            self.sim_to_screen(back1_x, back1_y),
            self.sim_to_screen(back2_x, back2_y)
        ]

    def _draw_frame_to_trail(self, population: pd.DataFrame) -> None:
        """Draw solid line trails using particle history, handling torus wrapping."""
        for idx, particle in population.iterrows():
            current_pos = (particle['x'], particle['y'])
            particle_color = (*self.get_color_for_whimsy(particle['whimsy']), 255)
            
            # Update particle history
            if idx not in self.particle_history:
                self.particle_history[idx] = []
            
            self.particle_history[idx].append(current_pos)
            if len(self.particle_history[idx]) > self.trail_length:
                self.particle_history[idx].pop(0)
            
            # Draw lines connecting historical positions
            positions = self.particle_history[idx]
            if len(positions) > 1:
                for i in range(len(positions) - 1):
                    pos1 = positions[i]
                    pos2 = positions[i + 1]
                    
                    # Check for wrap-around by looking at distance
                    dx = abs(pos2[0] - pos1[0])
                    dy = abs(pos2[1] - pos1[1])
                    
                    # If distance is less than 0.5 in both directions, it's not a wrap-around
                    if dx < 0.5 and dy < 0.5:
                        point1 = self.sim_to_screen(*pos1)
                        point2 = self.sim_to_screen(*pos2)
                        # Draw thick line first
                        pygame.draw.line(self._trail_surface, particle_color, point1, point2, width=self.trail_size)
                        # Draw thin anti-aliased line on top for smoothness
                        pygame.draw.aaline(self._trail_surface, particle_color, point1, point2)

    def _draw_current_particles(self, population: pd.DataFrame) -> None:
        """Draw triangular particles to the current surface."""
        self._current_surface.fill((0, 0, 0, 0))  # Clear current surface
        for _, particle in population.iterrows():
            particle_color = (*self.get_color_for_whimsy(particle['whimsy']), 255)
            triangle_points = self.calculate_triangle_points(
                particle['x'], 
                particle['y'], 
                particle['theta']
            )
            pygame.draw.polygon(self._current_surface, particle_color, triangle_points)

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

    def on_time_step(self, event: Event) -> None:
        """Update visualization each time step."""
        if self.scale is None:
            return
            
        pop = self.population_view.get(event.index)
        if pop.empty:
            return
        
        # Create and apply fade effect to trail surface
        fade_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        fade_surface.fill((0, 0, 0, int(255 * self.fade_speed)))
        self._trail_surface.blit(fade_surface, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)
        
        # Draw new trails
        self._draw_frame_to_trail(pop)
        
        # Draw current particle positions
        self._draw_current_particles(pop)
        
        # Compose final frame
        self._screen.fill(self.background_color)
        self._screen.blit(self._trail_surface, (0, 0))  # Draw trails first
        self._screen.blit(self._current_surface, (0, 0))  # Draw current particles on top
        self._draw_border()
        self._draw_progress_bar()
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key in [pygame.K_ESCAPE, pygame.K_q]
            ):
                pygame.quit()
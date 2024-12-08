import pygame
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData

class ParticleVisualizer(Component):
    """A component that visualizes particles with motion trails and collision trends."""
    
    @property
    def columns_required(self) -> Optional[List[str]]:
        return ['x', 'y', 'theta', 'whimsy', 'collision_count', 'last_collision_time']

    def __init__(self, 
                 background_color: tuple = (0, 0, 0),
                 progress_color: tuple = (0, 255, 0),
                 border_color: tuple = (100, 100, 100),
                 border_width: int = 2,
                 fade_speed: float = 0.02,
                 triangle_size: int = 6,
                 trail_size: int = 5,
                 trail_length: int = 5,
                 collision_color: tuple = (200, 0, 0),
                 max_collision_radius: int = 300,
                 collision_fade_speed: float = 0.05,
                 chart_size: tuple = (300, 150),  # Width and height of trend chart
                 fps: int = 12):
        super().__init__()
        self.background_color = background_color
        self.progress_color = progress_color
        self.border_color = border_color
        self.border_width = border_width
        self.fade_speed = fade_speed
        self.triangle_size = triangle_size
        self.trail_size = trail_size
        self.trail_length = trail_length
        self.collision_color = collision_color
        self.max_collision_radius = max_collision_radius
        self.collision_fade_speed = collision_fade_speed
        
        # Chart settings
        self.chart_size = chart_size
        self.chart_padding = 20  # Padding around the chart
        self.max_history_points = 100  # Number of time points to show
        self.max_collisions_shown = 10  # Will auto-adjust if exceeded
        
        # Dictionary to store previous positions
        self.particle_history = {}
        
        # Define color gradient endpoints
        self.low_whimsy_color = (41, 98, 255)    # Cool blue
        self.high_whimsy_color = (255, 89, 94)   # Warm red
        
        # Colors for different whimsy levels in the chart
        self.chart_colors = {
            'low': (41, 98, 255),     # Cool blue
            'medium': (147, 51, 234),  # Purple
            'high': (255, 89, 94)      # Warm red
        }
        
        # Initialize surfaces
        self._screen = None
        self._trail_surface = None
        self._current_surface = None
        self._collision_surface = None
        self._chart_surface = None
        
        self.scale = None
        self.view_offset = None
        self.simulation_bounds = None
        
        self.active_collisions = []
        
        # Initialize collision history
        self.collision_history = {
            'low': [],
            'medium': [],
            'high': []
        }
        
        self.fps = fps
        self.pygame_clock = pygame.time.Clock()  # Add pygame clock
        
    def setup(self, builder: Builder) -> None:
        pygame.init()
        
        screen_info = pygame.display.Info()
        self.width = screen_info.current_w
        self.height = screen_info.current_h
        self._screen = pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN)
        pygame.display.set_caption("Particle Simulation")
        
        self._trail_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self._trail_surface.fill((0, 0, 0, 0))
        
        self._current_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self._collision_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self._chart_surface = pygame.Surface(self.chart_size, pygame.SRCALPHA)
        
        self.start_time = pd.Timestamp(**builder.configuration.time.start)
        self.end_time = pd.Timestamp(**builder.configuration.time.end)
        self.clock = builder.time.clock()
        
        # Get reference to collisions component
        self.collisions = builder.components.get_component("collisions")
        
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

    def get_collision_trend_data(self, pop_index: pd.Index) -> None:
        pop = self.population_view.get(pop_index)
        current_time = self.clock()
        
        # Get collisions in current timestep
        current_collisions = pop[pop.last_collision_time == current_time]
        
        # Calculate counts by whimsy category
        whimsy_bins = pd.cut(current_collisions.whimsy, 
                            bins=[0, 1/3, 2/3, 1], 
                            labels=['low', 'medium', 'high'],
                            include_lowest=True)
        current_counts = whimsy_bins.value_counts()
        
        # Update history with cumulative counts
        for category in ['low', 'medium', 'high']:
            count = current_counts.get(category, 0)
            # Add to previous cumulative value or start at current count if no history
            new_value = count
            if self.collision_history[category]:
                new_value += self.collision_history[category][-1]
            self.collision_history[category].append(int(new_value))
            
            # Keep fixed number of points
            if len(self.collision_history[category]) > self.max_history_points:
                self.collision_history[category].pop(0)
    
    def _draw_frame_to_trail(self, population: pd.DataFrame) -> None:
        """Draw solid line trails using particle history, handling torus wrapping."""
        for idx, particle in population.iterrows():
            current_pos = (particle['x'], particle['y'])
            particle_color = (*self.get_color_for_whimsy(particle['whimsy']), 255)
            
            if idx not in self.particle_history:
                self.particle_history[idx] = []
            
            self.particle_history[idx].append(current_pos)
            if len(self.particle_history[idx]) > self.trail_length:
                self.particle_history[idx].pop(0)
            
            positions = self.particle_history[idx]
            if len(positions) > 1:
                for i in range(len(positions) - 1):
                    pos1 = positions[i]
                    pos2 = positions[i + 1]
                    
                    dx = abs(pos2[0] - pos1[0])
                    dy = abs(pos2[1] - pos1[1])
                    
                    if dx < 0.5 and dy < 0.5:
                        point1 = self.sim_to_screen(*pos1)
                        point2 = self.sim_to_screen(*pos2)
                        pygame.draw.line(self._trail_surface, particle_color, point1, point2, width=self.trail_size)
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
            
    def _draw_collisions(self) -> None:
        """Draw expanding circles at collision locations."""
        self._collision_surface.fill((0, 0, 0, 0))  # Clear collision surface
        
        # Add new collisions
        for x, y in self.collisions.current_collisions:
            self.active_collisions.append((x, y, 0))  # Age starts at 0
            
        # Update and draw existing collisions
        new_active_collisions = []
        for x, y, age in self.active_collisions:
            if age >= 1.0:  # Remove fully expanded collisions
                continue
                
            # Calculate current radius and alpha
            radius = int(age * self.max_collision_radius)
            alpha = int(255 * (1 - age))
            
            # Draw the collision circle
            screen_pos = self.sim_to_screen(x, y)
            color = (*self.collision_color, alpha)
            pygame.draw.circle(self._collision_surface, color, screen_pos, radius, 1)
            
            # Age the collision
            new_active_collisions.append((x, y, age + self.collision_fade_speed))
            
        self.active_collisions = new_active_collisions

    def _draw_trend_chart(self) -> None:
        """Draw the collision trend chart on its surface."""
        self._chart_surface.fill((0, 0, 0, 180))  # Semi-transparent background
        
        # Calculate dimensions
        chart_width = self.chart_size[0] - 2 * self.chart_padding
        chart_height = self.chart_size[1] - 2 * self.chart_padding
        
        # Find max value for scaling
        max_collisions = max(
            max(max(history) if history else 0 for history in self.collision_history.values()),
            1  # Prevent division by zero
        )
        
        # Draw axes
        pygame.draw.line(self._chart_surface, (200, 200, 200), 
                        (self.chart_padding, self.chart_size[1] - self.chart_padding),
                        (self.chart_size[0] - self.chart_padding, self.chart_size[1] - self.chart_padding))
        pygame.draw.line(self._chart_surface, (200, 200, 200),
                        (self.chart_padding, self.chart_padding),
                        (self.chart_padding, self.chart_size[1] - self.chart_padding))
        
        # Draw trend lines for each category
        for category, history in self.collision_history.items():
            if not history:
                continue
                
            points = []
            for i, value in enumerate(history):
                x = self.chart_padding + (i / (self.max_history_points - 1)) * chart_width
                y = (self.chart_size[1] - self.chart_padding - 
                    (value / max_collisions) * chart_height)
                points.append((int(x), int(y)))
            
            if len(points) > 1:
                pygame.draw.lines(self._chart_surface, self.chart_colors[category], False, points, 2)
        
        # Draw legend
        legend_y = self.chart_padding
        for category, color in self.chart_colors.items():
            pygame.draw.line(self._chart_surface, color, 
                        (self.chart_size[0] - 80, legend_y),
                        (self.chart_size[0] - 60, legend_y), 2)
            font = pygame.font.Font(None, 20)
            text = font.render(f"{category}: {self.collision_history[category][-1]}", True, color)
            self._chart_surface.blit(text, (self.chart_size[0] - 55, legend_y - 7))
            legend_y += 20
            
        # Draw scale with "Total Collisions" label
        font = pygame.font.Font(None, 20)
        max_label = font.render(str(max_collisions), True, (200, 200, 200))
        self._chart_surface.blit(max_label, (5, self.chart_padding - 10))
        zero_label = font.render('0', True, (200, 200, 200))
        self._chart_surface.blit(zero_label, (5, self.chart_size[1] - self.chart_padding - 10))
        
        # Add "Cumulative Collisions" title
        title = font.render('Cumulative Collisions', True, (200, 200, 200))
        self._chart_surface.blit(title, (self.chart_padding, 5))
    
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

    def _draw_fps(self) -> None:
        """Draw the current FPS in the corner of the screen."""
        font = pygame.font.Font(None, 24)
        fps_text = font.render(f'FPS: {int(self.pygame_clock.get_fps())}', True, (200, 200, 200))
        self._screen.blit(fps_text, (10, 10))

    def on_time_step(self, event: Event) -> None:
        """Update visualization each time step."""
        if self.scale is None:
            return
            
        pop = self.population_view.get(event.index)
        if pop.empty:
            return
            
        # Update collision trend data
        self.get_collision_trend_data(event.index)
        
        # Create and apply fade effect
        fade_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        fade_surface.fill((0, 0, 0, int(255 * self.fade_speed)))
        self._trail_surface.blit(fade_surface, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)
        
        # Draw all visualization elements
        self._draw_frame_to_trail(pop)
        self._draw_current_particles(pop)
        self._draw_collisions()
        self._draw_trend_chart()
        
        # Compose final frame
        self._screen.fill(self.background_color)
        self._screen.blit(self._trail_surface, (0, 0))
        self._screen.blit(self._collision_surface, (0, 0))
        self._screen.blit(self._current_surface, (0, 0))
        
        # Position chart in top-right corner with some padding
        chart_x = self.width - self.chart_size[0] - 20
        chart_y = 20
        self._screen.blit(self._chart_surface, (chart_x, chart_y))
        
        self._draw_border()
        self._draw_progress_bar()
        self._draw_fps()
        
        pygame.display.flip()
        
        # Control frame rate
        self.pygame_clock.tick(self.fps)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key in [pygame.K_ESCAPE, pygame.K_q]
            ):
                pygame.quit()
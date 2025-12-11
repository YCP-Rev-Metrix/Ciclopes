"""
Trajectory visualization for BEV lane analysis.

Generates publication-quality plots showing:
- Lane boundaries in BEV space
- Ball trajectory with different interpolation modes
- Error overlays and metrics
- BEV coordinate errors vs ground truth
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2

from .postprocessor import ProcessResult, LaneMetrics


class TrajectoryVisualizer:
    """
    Visualization tool for trajectory analysis in BEV space.
    
    BUG FIX #6: All plots use consistent Y-axis orientation (inverted to match
    image coordinates with origin at top-left). This ensures lane boundaries
    and trajectories align correctly across all visualization methods.
    """
    
    # Color scheme for clean, professional plots
    COLORS = {
        'lane': '#8B4513',  # Brown
        'trajectory_none': '#FF1744',  # Bright red
        'trajectory_linear': '#2196F3',  # Blue
        'trajectory_cubic': '#4CAF50',  # Green
        'start_marker': '#FFC107',  # Amber
        'end_marker': '#9C27B0',  # Purple
        'error_low': '#4CAF50',  # Green
        'error_med': '#FFC107',  # Amber
        'error_high': '#FF5252',  # Red
        'grid': '#E0E0E0',  # Light gray
        'text': '#212121',  # Dark gray
    }
    
    def __init__(self, bev_size: Tuple[int, int] = (400, 800)):
        """
        Initialize visualizer.
        
        Parameters
        ----------
        bev_size : Tuple[int, int]
            BEV output size (width, height) in pixels
        """
        self.bev_size = bev_size
        
    def extract_lane_boundary(self, warped_lane_mask: np.ndarray) -> np.ndarray:
        """
        Extract lane boundary polygon from BEV warped mask.
        
        Parameters
        ----------
        warped_lane_mask : np.ndarray
            Binary mask of lane in BEV space
            
        Returns
        -------
        np.ndarray
            Boundary points as (N, 2) array of (x, y) coordinates
        """
        if warped_lane_mask.ndim == 3:
            warped_lane_mask = warped_lane_mask.squeeze()
            
        # Clean up mask
        binary = (warped_lane_mask > 0).astype(np.uint8) * 255
        binary = cv2.morphologyEx(
            binary, 
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return np.array([])
            
        # Get largest contour (should be the lane)
        lane_contour = max(contours, key=cv2.contourArea)
        
        # Simplify polygon
        epsilon = 0.005 * cv2.arcLength(lane_contour, True)
        approx = cv2.approxPolyDP(lane_contour, epsilon, True)
        
        # Convert to (N, 2) array
        boundary = approx.reshape(-1, 2).astype(np.float64)
        
        return boundary
    
    def plot_single_trajectory(
        self,
        trajectory: List[Tuple[float, float]],
        lane_boundary: np.ndarray,
        interpolation_mode: str,
        errors: Optional[Dict[str, float]] = None,
        output_path: Optional[Path] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a single trajectory visualization.
        
        Parameters
        ----------
        trajectory : List[Tuple[float, float]]
            List of (x, y) centroid positions in BEV space
        lane_boundary : np.ndarray
            Lane boundary points as (N, 2) array
        interpolation_mode : str
            Interpolation mode: "none", "linear", or "cubic"
        errors : Optional[Dict[str, float]]
            Error metrics to display
        output_path : Optional[Path]
            If provided, save figure to this path
        title : Optional[str]
            Plot title
            
        Returns
        -------
        plt.Figure
            The generated figure
        """
        fig, ax = plt.subplots(figsize=(8, 12), dpi=150)
        
        # Set background color
        ax.set_facecolor('#FAFAFA')
        fig.patch.set_facecolor('white')
        
        # Plot lane boundary
        if lane_boundary.size > 0:
            lane_x = lane_boundary[:, 0]
            lane_y = lane_boundary[:, 1]
            ax.fill(
                lane_x, lane_y, 
                color=self.COLORS['lane'], 
                alpha=0.3, 
                label='Lane Boundary'
            )
            ax.plot(
                lane_x, lane_y, 
                color=self.COLORS['lane'], 
                linewidth=2, 
                linestyle='--'
            )
        
        # Plot trajectory
        if trajectory:
            traj_array = np.array(trajectory)
            traj_x = traj_array[:, 0]
            traj_y = traj_array[:, 1]
            
            # Choose color based on interpolation mode
            color_map = {
                'none': self.COLORS['trajectory_none'],
                'linear': self.COLORS['trajectory_linear'],
                'cubic': self.COLORS['trajectory_cubic'],
            }
            traj_color = color_map.get(interpolation_mode, self.COLORS['trajectory_none'])
            
            # Plot trajectory line
            ax.plot(
                traj_x, traj_y, 
                color=traj_color, 
                linewidth=3, 
                label=f'Trajectory ({interpolation_mode})',
                marker='o',
                markersize=4,
                markerfacecolor=traj_color,
                markeredgecolor='white',
                markeredgewidth=0.5,
                alpha=0.9
            )
            
            # Mark start and end points
            ax.scatter(
                traj_x[0], traj_y[0], 
                s=200, 
                color=self.COLORS['start_marker'], 
                edgecolor='white',
                linewidth=2,
                marker='o', 
                label='Start',
                zorder=5
            )
            ax.scatter(
                traj_x[-1], traj_y[-1], 
                s=200, 
                color=self.COLORS['end_marker'], 
                edgecolor='white',
                linewidth=2,
                marker='s', 
                label='End',
                zorder=5
            )
            
            # Add direction arrows
            n_arrows = 5
            arrow_indices = np.linspace(0, len(traj_x) - 1, n_arrows + 2, dtype=int)[1:-1]
            for idx in arrow_indices:
                if idx < len(traj_x) - 1:
                    dx = traj_x[idx + 1] - traj_x[idx]
                    dy = traj_y[idx + 1] - traj_y[idx]
                    ax.arrow(
                        traj_x[idx], traj_y[idx], dx * 0.8, dy * 0.8,
                        head_width=8, head_length=10, 
                        fc=traj_color, ec=traj_color, 
                        alpha=0.6, zorder=4
                    )
        
        # Add error text box
        if errors:
            error_text = self._format_error_text(errors, interpolation_mode)
            props = dict(boxstyle='round,pad=0.8', facecolor='white', 
                        edgecolor=self.COLORS['text'], alpha=0.95, linewidth=1.5)
            ax.text(
                0.02, 0.98, error_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=props,
                family='monospace',
                color=self.COLORS['text']
            )
        
        # Formatting
        ax.set_xlabel('BEV X (pixels)', fontsize=12, fontweight='bold')
        ax.set_ylabel('BEV Y (pixels)', fontsize=12, fontweight='bold')
        
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        else:
            mode_title = interpolation_mode.capitalize()
            ax.set_title(
                f'Ball Trajectory - {mode_title} Interpolation', 
                fontsize=14, fontweight='bold', pad=20
            )
        
        ax.legend(loc='upper right', fontsize=10, framealpha=0.95, edgecolor=self.COLORS['text'])
        ax.grid(True, alpha=0.3, color=self.COLORS['grid'], linestyle='-', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')
        
        # Invert y-axis to match image coordinates
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Saved trajectory plot: {output_path}")
        
        return fig
    
    def plot_averaged_trajectory(
        self,
        trajectories: Dict[str, List[Tuple[float, float]]],
        lane_boundary: np.ndarray,
        errors: Dict[str, Dict[str, float]],
        output_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Create clean averaged trajectory plot across all interpolation modes.

        Parameters
        ----------
        trajectories : Dict[str, List[Tuple[float, float]]]
            Trajectories for each interpolation mode
        lane_boundary : np.ndarray
            Lane boundary points
        errors : Dict[str, Dict[str, float]]
            Error metrics for each mode (will be averaged)
        output_path : Optional[Path]
            If provided, save figure to this path

        Returns
        -------
        plt.Figure
            The generated figure
        """
        fig, ax = plt.subplots(figsize=(10, 14), dpi=150)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#F5F5F5')

        # Plot lane boundary
        if lane_boundary.size > 0:
            lane_x = lane_boundary[:, 0]
            lane_y = lane_boundary[:, 1]
            ax.fill(lane_x, lane_y, color=self.COLORS['lane'], alpha=0.15, label='Lane')
            ax.plot(lane_x, lane_y, color=self.COLORS['lane'],
                   linewidth=2.5, linestyle='-', alpha=0.8)

        # Compute averaged trajectory
        modes = ['none', 'linear', 'cubic']
        valid_trajectories = [np.array(trajectories[m]) for m in modes if m in trajectories and trajectories[m]]

        if valid_trajectories:
            # Find minimum length to align trajectories
            min_len = min(len(t) for t in valid_trajectories)
            aligned_trajectories = [t[:min_len] for t in valid_trajectories]

            # Compute mean and std
            traj_stack = np.stack(aligned_trajectories, axis=0)
            traj_mean = np.mean(traj_stack, axis=0)
            traj_std = np.std(traj_stack, axis=0)

            # Extract coordinates
            x_mean = traj_mean[:, 0]
            y_mean = traj_mean[:, 1]
            x_std = traj_std[:, 0]
            y_std = traj_std[:, 1]

            # Plot confidence band around trajectory
            for i in range(len(x_mean) - 1):
                ellipse = mpatches.Ellipse(
                    (x_mean[i], y_mean[i]),
                    width=2 * x_std[i],
                    height=2 * y_std[i],
                    alpha=0.15,
                    color='#2196F3',
                    zorder=1
                )
                ax.add_patch(ellipse)
            # Add label for legend
            ax.plot([], [], color='#2196F3', alpha=0.3, linewidth=10, label='±1 σ')

            # Plot mean trajectory
            ax.plot(
                x_mean, y_mean,
                color='#1565C0',
                linewidth=4,
                label='Mean Trajectory',
                zorder=3
            )

            # Plot individual ball positions
            ax.scatter(
                x_mean, y_mean,
                s=80,
                color='#1976D2',
                edgecolor='white',
                linewidth=1.5,
                alpha=0.7,
                zorder=4
            )

            # Start marker
            ax.scatter(
                x_mean[0], y_mean[0],
                s=300,
                color='#4CAF50',
                edgecolor='white',
                linewidth=3,
                marker='o',
                label='Start',
                zorder=6
            )

            # End marker
            ax.scatter(
                x_mean[-1], y_mean[-1],
                s=300,
                color='#F44336',
                edgecolor='white',
                linewidth=3,
                marker='s',
                label='End',
                zorder=6
            )

            # Add direction arrows (fewer, cleaner)
            n_arrows = 3
            arrow_indices = np.linspace(0, len(x_mean) - 1, n_arrows + 2, dtype=int)[1:-1]
            for idx in arrow_indices:
                if idx < len(x_mean) - 1:
                    dx = x_mean[idx + 1] - x_mean[idx]
                    dy = y_mean[idx + 1] - y_mean[idx]
                    ax.arrow(
                        x_mean[idx], y_mean[idx], dx * 0.7, dy * 0.7,
                        head_width=12, head_length=15,
                        fc='#1565C0', ec='#1565C0',
                        alpha=0.5, zorder=5, linewidth=2
                    )

        # Compute averaged errors
        avg_errors = {}
        for key in ['speed_error', 'accel_mag_error', 'total_break_error', 'end_speed_error', 'seg_map_50_95']:
            values = [errors[m].get(key, 0) for m in modes if m in errors]
            if values:
                avg_errors[key] = np.mean(values)

        # Add clean error summary
        if avg_errors:
            error_lines = ['Performance Metrics', '─' * 22]
            if 'speed_error' in avg_errors:
                error_lines.append(f'Speed MAE:     {abs(avg_errors["speed_error"]):6.2f} m/s')
            if 'accel_mag_error' in avg_errors:
                error_lines.append(f'Accel MAE:     {abs(avg_errors["accel_mag_error"]):6.2f} m/s²')
            if 'total_break_error' in avg_errors:
                error_lines.append(f'Break Error:   {abs(avg_errors["total_break_error"]):6.3f} m')
            if 'seg_map_50_95' in avg_errors:
                error_lines.append('─' * 22)
                error_lines.append(f'mAP@50-95:     {avg_errors["seg_map_50_95"]:6.1%}')

            error_text = '\n'.join(error_lines)
            props = dict(boxstyle='round,pad=0.8', facecolor='white',
                        edgecolor='#424242', alpha=0.95, linewidth=2)
            ax.text(
                0.03, 0.97, error_text,
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment='top',
                bbox=props,
                family='monospace',
                color='#212121'
            )

        # Clean formatting
        ax.set_xlabel('BEV X Coordinate (pixels)', fontsize=13, fontweight='bold', color='#212121')
        ax.set_ylabel('BEV Y Coordinate (pixels)', fontsize=13, fontweight='bold', color='#212121')
        ax.set_title('Ball Trajectory on Lane (Averaged)', fontsize=16, fontweight='bold', pad=20, color='#212121')

        ax.legend(loc='lower right', fontsize=11, framealpha=0.95,
                 edgecolor='#424242', fancybox=True, shadow=False)
        ax.grid(True, alpha=0.25, color='#BDBDBD', linestyle='-', linewidth=0.8)
        ax.set_aspect('equal', adjustable='box')

        # BUG FIX #6: Standardize Y-axis orientation across all plots
        # Invert Y to match image coordinates (origin at top-left)
        # This is consistent with plot_single_trajectory and plot_comparison
        ax.invert_yaxis()

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Saved averaged trajectory plot: {output_path}")

        return fig

    def plot_comparison(
        self,
        trajectories: Dict[str, List[Tuple[float, float]]],
        lane_boundary: np.ndarray,
        errors: Dict[str, Dict[str, float]],
        output_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Create side-by-side comparison of all interpolation methods.

        Parameters
        ----------
        trajectories : Dict[str, List[Tuple[float, float]]]
            Trajectories for each interpolation mode
        lane_boundary : np.ndarray
            Lane boundary points
        errors : Dict[str, Dict[str, float]]
            Error metrics for each mode
        output_path : Optional[Path]
            If provided, save figure to this path

        Returns
        -------
        plt.Figure
            The generated figure
        """
        fig = plt.figure(figsize=(18, 12), dpi=150)
        fig.patch.set_facecolor('white')

        modes = ['none', 'linear', 'cubic']
        mode_titles = ['No Interpolation', 'Piecewise Linear', 'Cubic Spline']

        for idx, (mode, mode_title) in enumerate(zip(modes, mode_titles)):
            ax = fig.add_subplot(1, 3, idx + 1)
            ax.set_facecolor('#FAFAFA')

            # Plot lane boundary
            if lane_boundary.size > 0:
                lane_x = lane_boundary[:, 0]
                lane_y = lane_boundary[:, 1]
                ax.fill(lane_x, lane_y, color=self.COLORS['lane'], alpha=0.3)
                ax.plot(lane_x, lane_y, color=self.COLORS['lane'],
                       linewidth=2, linestyle='--', label='Lane')

            # Plot trajectory if available
            if mode in trajectories and trajectories[mode]:
                traj_array = np.array(trajectories[mode])
                traj_x = traj_array[:, 0]
                traj_y = traj_array[:, 1]

                color_map = {
                    'none': self.COLORS['trajectory_none'],
                    'linear': self.COLORS['trajectory_linear'],
                    'cubic': self.COLORS['trajectory_cubic'],
                }
                traj_color = color_map.get(mode, self.COLORS['trajectory_none'])

                ax.plot(traj_x, traj_y, color=traj_color, linewidth=3,
                       marker='o', markersize=4, markerfacecolor=traj_color,
                       markeredgecolor='white', markeredgewidth=0.5,
                       alpha=0.9, label='Trajectory')

                # Start/end markers
                ax.scatter(traj_x[0], traj_y[0], s=200,
                          color=self.COLORS['start_marker'],
                          edgecolor='white', linewidth=2, marker='o',
                          label='Start', zorder=5)
                ax.scatter(traj_x[-1], traj_y[-1], s=200,
                          color=self.COLORS['end_marker'],
                          edgecolor='white', linewidth=2, marker='s',
                          label='End', zorder=5)

            # Add error text
            if mode in errors:
                error_text = self._format_error_text(errors[mode], mode)
                props = dict(boxstyle='round,pad=0.6', facecolor='white',
                           edgecolor=self.COLORS['text'], alpha=0.95, linewidth=1.5)
                ax.text(0.02, 0.98, error_text, transform=ax.transAxes,
                       fontsize=9, verticalalignment='top', bbox=props,
                       family='monospace', color=self.COLORS['text'])

            # Formatting
            ax.set_xlabel('BEV X (pixels)', fontsize=11, fontweight='bold')
            if idx == 0:
                ax.set_ylabel('BEV Y (pixels)', fontsize=11, fontweight='bold')
            ax.set_title(mode_title, fontsize=13, fontweight='bold', pad=15)
            ax.legend(loc='upper right', fontsize=9, framealpha=0.95)
            ax.grid(True, alpha=0.3, color=self.COLORS['grid'])
            ax.set_aspect('equal', adjustable='box')
            ax.invert_yaxis()

        fig.suptitle('Trajectory Comparison: Interpolation Methods',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Saved comparison plot: {output_path}")

        return fig
    
    def _format_error_text(self, errors: Dict[str, float], mode: str) -> str:
        """Format error metrics as clean text for overlay."""
        lines = [f"Mode: {mode.upper()}"]
        lines.append("─" * 25)
        
        # Velocity errors (lane coordinates: s=along-lane, l=lateral)
        if 'vel_s_error' in errors:
            lines.append(f"Lane Vel Err: {errors['vel_s_error']:>7.2f} m/s")
        if 'vel_l_error' in errors:
            lines.append(f"Lat Vel Err:  {errors['vel_l_error']:>7.2f} m/s")
        if 'speed_error' in errors:
            lines.append(f"Speed Error:  {errors['speed_error']:>7.2f} m/s")
        
        # Acceleration errors
        if 'accel_mag_error' in errors:
            lines.append(f"Accel Error:  {errors['accel_mag_error']:>7.2f} m/s²")
        
        # Position errors
        if 'total_break_error' in errors:
            lines.append(f"Break Error:  {errors['total_break_error']:>7.3f} m")
        if 'end_speed_error' in errors:
            lines.append(f"End Spd Err:  {errors['end_speed_error']:>7.2f} m/s")
        
        # BEV coordinate errors (new)
        if 'bev_x_rmse' in errors:
            lines.append("─" * 25)
            lines.append(f"BEV X RMSE:   {errors['bev_x_rmse']:>7.2f} px")
        if 'bev_y_rmse' in errors:
            lines.append(f"BEV Y RMSE:   {errors['bev_y_rmse']:>7.2f} px")
        if 'bev_total_rmse' in errors:
            lines.append(f"BEV Tot RMSE: {errors['bev_total_rmse']:>7.2f} px")
        
        # Detection quality
        if 'seg_map_50_95' in errors:
            lines.append("─" * 25)
            lines.append(f"mAP@50-95:    {errors['seg_map_50_95']:>7.1%}")
        
        return '\n'.join(lines)
    
    def plot_single_mode_trajectory(
        self,
        trajectory: List[Tuple[float, float]],
        gt_trajectory: Optional[List[Tuple[float, float]]],
        lane_boundary: np.ndarray,
        mode: str,
        errors: Dict[str, float],
        bev_errors: Dict[str, float],
        output_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Create a single trajectory visualization for one interpolation mode.
        
        Layout: Lane plot on left (large), metrics table on right.
        
        Parameters
        ----------
        trajectory : List[Tuple[float, float]]
            Trajectory points for this mode
        gt_trajectory : Optional[List[Tuple[float, float]]]
            Ground truth trajectory in BEV coordinates
        lane_boundary : np.ndarray
            Lane boundary points as (N, 2) array
        mode : str
            Interpolation mode: "none", "linear", or "cubic"
        errors : Dict[str, float]
            World/lane coordinate error metrics
        bev_errors : Dict[str, float]
            BEV coordinate error metrics
        output_path : Optional[Path]
            If provided, save figure to this path
            
        Returns
        -------
        plt.Figure
            The generated figure
        """
        mode_titles = {
            'none': 'No Interpolation',
            'linear': 'Linear Interpolation', 
            'cubic': 'Cubic Spline',
        }
        mode_colors = {
            'none': self.COLORS['trajectory_none'],
            'linear': self.COLORS['trajectory_linear'],
            'cubic': self.COLORS['trajectory_cubic'],
        }
        
        # Create figure with lane on left, metrics on right
        fig = plt.figure(figsize=(16, 10), dpi=150)
        fig.patch.set_facecolor('#FAFAFA')
        
        # GridSpec: lane plot takes 70%, metrics table takes 30%
        gs = GridSpec(1, 2, figure=fig, width_ratios=[2.5, 1], wspace=0.05)
        
        # Left panel: Lane and trajectory
        ax_lane = fig.add_subplot(gs[0, 0])
        self._plot_lane_panel(
            ax=ax_lane,
            trajectory=trajectory,
            gt_trajectory=gt_trajectory,
            lane_boundary=lane_boundary,
            color=mode_colors.get(mode, self.COLORS['trajectory_none']),
        )
        
        # Right panel: Metrics table
        ax_metrics = fig.add_subplot(gs[0, 1])
        self._plot_metrics_panel(
            ax=ax_metrics,
            mode=mode,
            errors=errors,
            bev_errors=bev_errors,
        )
        
        fig.suptitle(
            f'Ball Trajectory Analysis: {mode_titles.get(mode, mode.upper())}',
            fontsize=16, fontweight='bold', y=0.98, color='#212121'
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
            print(f"Saved {mode} trajectory plot: {output_path}")
        
        return fig
    
    def plot_overlay_trajectory(
        self,
        trajectories: Dict[str, List[Tuple[float, float]]],
        gt_trajectory: Optional[List[Tuple[float, float]]],
        lane_boundary: np.ndarray,
        errors_by_mode: Dict[str, Dict[str, float]],
        bev_errors_by_mode: Dict[str, Dict[str, float]],
        output_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Create overlay plot showing all interpolation modes together.
        
        Layout: Lane plot with all trajectories on left, comparison table on right.
        
        Parameters
        ----------
        trajectories : Dict[str, List[Tuple[float, float]]]
            Trajectories for each interpolation mode
        gt_trajectory : Optional[List[Tuple[float, float]]]
            Ground truth trajectory in BEV coordinates
        lane_boundary : np.ndarray
            Lane boundary points as (N, 2) array
        errors_by_mode : Dict[str, Dict[str, float]]
            World/lane coordinate error metrics for each mode
        bev_errors_by_mode : Dict[str, Dict[str, float]]
            BEV coordinate error metrics for each mode
        output_path : Optional[Path]
            If provided, save figure to this path
            
        Returns
        -------
        plt.Figure
            The generated figure
        """
        mode_colors = {
            'none': self.COLORS['trajectory_none'],
            'linear': self.COLORS['trajectory_linear'],
            'cubic': self.COLORS['trajectory_cubic'],
        }
        
        # Create figure with lane on left, comparison table on right
        fig = plt.figure(figsize=(18, 10), dpi=150)
        fig.patch.set_facecolor('#FAFAFA')
        
        # GridSpec: lane plot takes 65%, metrics table takes 35%
        gs = GridSpec(1, 2, figure=fig, width_ratios=[2, 1.2], wspace=0.08)
        
        # Left panel: Lane with all trajectories overlaid
        ax_lane = fig.add_subplot(gs[0, 0])
        self._plot_overlay_lane_panel(
            ax=ax_lane,
            trajectories=trajectories,
            gt_trajectory=gt_trajectory,
            lane_boundary=lane_boundary,
            mode_colors=mode_colors,
        )
        
        # Right panel: Comparison metrics table
        ax_metrics = fig.add_subplot(gs[0, 1])
        self._plot_comparison_metrics_panel(
            ax=ax_metrics,
            errors_by_mode=errors_by_mode,
            bev_errors_by_mode=bev_errors_by_mode,
        )
        
        fig.suptitle(
            'Ball Trajectory Analysis: All Modes Comparison',
            fontsize=16, fontweight='bold', y=0.98, color='#212121'
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#FAFAFA')
            print(f"Saved overlay trajectory plot: {output_path}")
        
        return fig
    
    def _plot_lane_panel(
        self,
        ax: plt.Axes,
        trajectory: List[Tuple[float, float]],
        gt_trajectory: Optional[List[Tuple[float, float]]],
        lane_boundary: np.ndarray,
        color: str,
    ) -> None:
        """Plot lane with single trajectory."""
        ax.set_facecolor('#F5F5F5')
        
        # Plot lane boundary (filled region)
        if lane_boundary.size > 0:
            lane_x = lane_boundary[:, 0]
            lane_y = lane_boundary[:, 1]
            ax.fill(lane_x, lane_y, color=self.COLORS['lane'], alpha=0.25, label='Lane')
            ax.plot(lane_x, lane_y, color=self.COLORS['lane'], linewidth=2.5, 
                   linestyle='-', alpha=0.8)
        
        # Note: GT trajectory not shown in BEV space because inverse calibration
        # is only approximate. BEV errors are still computed and displayed in metrics.
        
        # Plot predicted trajectory with ball positions
        if trajectory:
            traj_arr = np.array(trajectory)
            traj_x = traj_arr[:, 0]
            traj_y = traj_arr[:, 1]
            
            # Trajectory line
            ax.plot(traj_x, traj_y, color=color, linewidth=3, alpha=0.8, zorder=3, 
                   label='Predicted')
            
            # Ball position markers
            ax.scatter(traj_x, traj_y, s=80, c=color, edgecolor='white', 
                      linewidth=1.5, alpha=0.9, zorder=4)
            
            # Start marker
            ax.scatter(traj_x[0], traj_y[0], s=300, c='#4CAF50', edgecolor='white',
                      linewidth=3, marker='o', label='Start', zorder=6)
            
            # End marker
            ax.scatter(traj_x[-1], traj_y[-1], s=300, c='#F44336', edgecolor='white',
                      linewidth=3, marker='s', label='End', zorder=6)
            
            # Direction arrows
            n_arrows = min(5, len(traj_x) - 1)
            if n_arrows > 0:
                arrow_indices = np.linspace(0, len(traj_x) - 2, n_arrows, dtype=int)
                for i in arrow_indices:
                    dx = traj_x[i + 1] - traj_x[i]
                    dy = traj_y[i + 1] - traj_y[i]
                    ax.arrow(traj_x[i], traj_y[i], dx * 0.6, dy * 0.6,
                            head_width=12, head_length=10, fc=color, ec=color,
                            alpha=0.5, zorder=5)
        
        # Formatting
        ax.set_xlabel('BEV X (pixels)', fontsize=12, fontweight='bold')
        ax.set_ylabel('BEV Y (pixels)', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3, color='#BDBDBD', linestyle='-', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')
        ax.invert_yaxis()
    
    def _plot_overlay_lane_panel(
        self,
        ax: plt.Axes,
        trajectories: Dict[str, List[Tuple[float, float]]],
        gt_trajectory: Optional[List[Tuple[float, float]]],
        lane_boundary: np.ndarray,
        mode_colors: Dict[str, str],
    ) -> None:
        """Plot lane with all trajectories overlaid."""
        ax.set_facecolor('#F5F5F5')
        
        # Plot lane boundary
        if lane_boundary.size > 0:
            lane_x = lane_boundary[:, 0]
            lane_y = lane_boundary[:, 1]
            ax.fill(lane_x, lane_y, color=self.COLORS['lane'], alpha=0.2, label='Lane')
            ax.plot(lane_x, lane_y, color=self.COLORS['lane'], linewidth=3, 
                   linestyle='-', alpha=0.8)
        
        # Note: GT trajectory not shown in BEV space because inverse calibration
        # is only approximate. BEV errors are still computed and displayed in metrics.
        
        modes = ['none', 'linear', 'cubic']
        mode_labels = {'none': 'None', 'linear': 'Linear', 'cubic': 'Cubic'}
        
        # Plot each trajectory
        for mode in modes:
            traj = trajectories.get(mode, [])
            if not traj:
                continue
            
            traj_arr = np.array(traj)
            color = mode_colors[mode]
            
            ax.plot(traj_arr[:, 0], traj_arr[:, 1], color=color, linewidth=2.5,
                   alpha=0.8, label=f'{mode_labels[mode]}', zorder=3)
            ax.scatter(traj_arr[:, 0], traj_arr[:, 1], s=50, c=color, 
                      edgecolor='white', linewidth=1, alpha=0.7, zorder=4)
        
        # Shared start/end markers from first available trajectory
        for mode in modes:
            traj = trajectories.get(mode, [])
            if traj:
                traj_arr = np.array(traj)
                ax.scatter(traj_arr[0, 0], traj_arr[0, 1], s=350, c='#4CAF50', 
                          edgecolor='white', linewidth=3, marker='o', 
                          label='Start', zorder=6)
                ax.scatter(traj_arr[-1, 0], traj_arr[-1, 1], s=350, c='#F44336', 
                          edgecolor='white', linewidth=3, marker='s', 
                          label='End', zorder=6)
                break
        
        # Formatting
        ax.set_xlabel('BEV X (pixels)', fontsize=12, fontweight='bold')
        ax.set_ylabel('BEV Y (pixels)', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10, framealpha=0.95, ncol=2)
        ax.grid(True, alpha=0.3, color='#BDBDBD', linestyle='-', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')
        ax.invert_yaxis()
    
    def _plot_metrics_panel(
        self,
        ax: plt.Axes,
        mode: str,
        errors: Dict[str, float],
        bev_errors: Dict[str, float],
    ) -> None:
        """Plot metrics as a clean table panel."""
        ax.set_facecolor('#FAFAFA')
        ax.axis('off')
        
        mode_titles = {'none': 'NO INTERPOLATION', 'linear': 'LINEAR', 'cubic': 'CUBIC SPLINE'}
        
        # Build table data
        table_data = []
        row_colors = []
        
        # Header section
        table_data.append(['WORLD METRICS', ''])
        row_colors.append('#E3F2FD')
        
        if 'speed_error' in errors:
            table_data.append(['Speed MAE', f"{abs(errors['speed_error']):.3f} m/s"])
            row_colors.append('white')
        if 'accel_mag_error' in errors:
            table_data.append(['Accel MAE', f"{abs(errors['accel_mag_error']):.3f} m/s²"])
            row_colors.append('white')
        if 'total_break_error' in errors:
            table_data.append(['Break Error', f"{abs(errors['total_break_error']):.4f} m"])
            row_colors.append('white')
        if 'end_speed_error' in errors:
            table_data.append(['End Speed Err', f"{abs(errors['end_speed_error']):.3f} m/s"])
            row_colors.append('white')
        
        # BEV section
        table_data.append(['BEV ERRORS', ''])
        row_colors.append('#E8F5E9')
        
        if 'bev_x_rmse' in bev_errors:
            table_data.append(['X RMSE', f"{bev_errors['bev_x_rmse']:.2f} px"])
            row_colors.append('white')
        if 'bev_y_rmse' in bev_errors:
            table_data.append(['Y RMSE', f"{bev_errors['bev_y_rmse']:.2f} px"])
            row_colors.append('white')
        if 'bev_total_rmse' in bev_errors:
            table_data.append(['Total RMSE', f"{bev_errors['bev_total_rmse']:.2f} px"])
            row_colors.append('white')
        
        # Detection quality
        table_data.append(['DETECTION', ''])
        row_colors.append('#FFF3E0')
        
        if 'seg_map_50_95' in errors:
            table_data.append(['mAP@50-95', f"{errors['seg_map_50_95']:.1%}"])
            row_colors.append('white')
        
        # Create table
        table = ax.table(
            cellText=table_data,
            cellLoc='left',
            loc='center',
            cellColours=[[c, c] for c in row_colors],
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.0, 1.8)
        
        # Style header cells
        for i, row in enumerate(table_data):
            if row[1] == '':  # Section headers
                table[(i, 0)].set_text_props(fontweight='bold', fontsize=12)
                table[(i, 1)].set_text_props(fontweight='bold')
        
        # Add title
        ax.set_title(f'{mode_titles.get(mode, mode.upper())}', 
                    fontsize=14, fontweight='bold', pad=20, color='#212121')
    
    def _plot_comparison_metrics_panel(
        self,
        ax: plt.Axes,
        errors_by_mode: Dict[str, Dict[str, float]],
        bev_errors_by_mode: Dict[str, Dict[str, float]],
    ) -> None:
        """Plot comparison metrics table for all modes."""
        ax.set_facecolor('#FAFAFA')
        ax.axis('off')
        
        modes = ['none', 'linear', 'cubic']
        
        # Build comparison table
        table_data = []
        row_colors = []
        
        # Header row
        table_data.append(['Metric', 'None', 'Linear', 'Cubic', 'Avg'])
        row_colors.append('#E0E0E0')
        
        # Speed MAE
        vals = [abs(errors_by_mode.get(m, {}).get('speed_error', 0)) for m in modes]
        avg = np.mean(vals)
        table_data.append(['Speed (m/s)', f'{vals[0]:.2f}', f'{vals[1]:.2f}', f'{vals[2]:.2f}', f'{avg:.2f}'])
        row_colors.append('#E3F2FD')
        
        # Accel MAE
        vals = [abs(errors_by_mode.get(m, {}).get('accel_mag_error', 0)) for m in modes]
        avg = np.mean(vals)
        table_data.append(['Accel (m/s²)', f'{vals[0]:.2f}', f'{vals[1]:.2f}', f'{vals[2]:.2f}', f'{avg:.2f}'])
        row_colors.append('white')
        
        # Break Error
        vals = [abs(errors_by_mode.get(m, {}).get('total_break_error', 0)) for m in modes]
        avg = np.mean(vals)
        table_data.append(['Break (m)', f'{vals[0]:.4f}', f'{vals[1]:.4f}', f'{vals[2]:.4f}', f'{avg:.4f}'])
        row_colors.append('#E3F2FD')
        
        # End Speed
        vals = [abs(errors_by_mode.get(m, {}).get('end_speed_error', 0)) for m in modes]
        avg = np.mean(vals)
        table_data.append(['End Spd (m/s)', f'{vals[0]:.2f}', f'{vals[1]:.2f}', f'{vals[2]:.2f}', f'{avg:.2f}'])
        row_colors.append('white')
        
        # Separator
        table_data.append(['', '', '', '', ''])
        row_colors.append('#FAFAFA')
        
        # BEV X RMSE
        vals = [bev_errors_by_mode.get(m, {}).get('bev_x_rmse', 0) for m in modes]
        avg = np.mean(vals)
        table_data.append(['BEV X (px)', f'{vals[0]:.2f}', f'{vals[1]:.2f}', f'{vals[2]:.2f}', f'{avg:.2f}'])
        row_colors.append('#E8F5E9')
        
        # BEV Y RMSE
        vals = [bev_errors_by_mode.get(m, {}).get('bev_y_rmse', 0) for m in modes]
        avg = np.mean(vals)
        table_data.append(['BEV Y (px)', f'{vals[0]:.2f}', f'{vals[1]:.2f}', f'{vals[2]:.2f}', f'{avg:.2f}'])
        row_colors.append('white')
        
        # BEV Total RMSE
        vals = [bev_errors_by_mode.get(m, {}).get('bev_total_rmse', 0) for m in modes]
        avg = np.mean(vals)
        table_data.append(['BEV Total (px)', f'{vals[0]:.2f}', f'{vals[1]:.2f}', f'{vals[2]:.2f}', f'{avg:.2f}'])
        row_colors.append('#E8F5E9')
        
        # Separator
        table_data.append(['', '', '', '', ''])
        row_colors.append('#FAFAFA')
        
        # mAP
        vals = [errors_by_mode.get(m, {}).get('seg_map_50_95', 0) for m in modes]
        avg = np.mean(vals)
        table_data.append(['mAP@50-95', f'{vals[0]:.1%}', f'{vals[1]:.1%}', f'{vals[2]:.1%}', f'{avg:.1%}'])
        row_colors.append('#FFF3E0')
        
        # Create table
        table = ax.table(
            cellText=table_data,
            cellLoc='center',
            loc='center',
            cellColours=[[c] * 5 for c in row_colors],
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.0, 1.6)
        
        # Style header row
        for j in range(5):
            table[(0, j)].set_text_props(fontweight='bold', fontsize=12)
        
        # Style first column
        for i in range(len(table_data)):
            table[(i, 0)].set_text_props(fontweight='bold')
        
        ax.set_title('Metrics Comparison', fontsize=14, fontweight='bold', pad=20, color='#212121')
    
    def _format_metrics_key(
        self, 
        mode: str, 
        errors: Dict[str, float], 
        bev_errors: Dict[str, float]
    ) -> str:
        """Format a compact metrics key for a single panel."""
        lines = [f"═══ {mode.upper()} ═══"]
        
        # World/Lane metrics
        if errors:
            lines.append("World Metrics:")
            if 'speed_error' in errors:
                lines.append(f"  Speed MAE:  {abs(errors['speed_error']):6.2f} m/s")
            if 'accel_mag_error' in errors:
                lines.append(f"  Accel MAE:  {abs(errors['accel_mag_error']):6.2f} m/s²")
            if 'total_break_error' in errors:
                lines.append(f"  Break Err:  {abs(errors['total_break_error']):6.3f} m")
            if 'end_speed_error' in errors:
                lines.append(f"  End Spd:    {abs(errors['end_speed_error']):6.2f} m/s")
        
        # BEV coordinate errors
        if bev_errors:
            lines.append("BEV Errors:")
            if 'bev_x_rmse' in bev_errors:
                lines.append(f"  X RMSE:     {bev_errors['bev_x_rmse']:6.2f} px")
            if 'bev_y_rmse' in bev_errors:
                lines.append(f"  Y RMSE:     {bev_errors['bev_y_rmse']:6.2f} px")
            if 'bev_total_rmse' in bev_errors:
                lines.append(f"  Total RMSE: {bev_errors['bev_total_rmse']:6.2f} px")
        
        # Detection quality
        if 'seg_map_50_95' in errors:
            lines.append(f"mAP@50-95:    {errors['seg_map_50_95']:6.1%}")
        
        return '\n'.join(lines)
    
    def extract_trajectory_from_results(
        self, 
        results_by_index: Dict[int, ProcessResult],
        apply_interpolation: bool = False,
        interpolation_mode: str = "none",
        dt: float = 1.0 / 30.0,
    ) -> List[Tuple[float, float]]:
        """
        Extract trajectory points from PostProcessor results.
        
        BUG FIX #5: Added option to apply interpolation so visualization matches
        the data used for metrics computation.
        
        Parameters
        ----------
        results_by_index : Dict[int, ProcessResult]
            Results dictionary from PostProcessor.process_run()
        apply_interpolation : bool
            If True, apply interpolation to match metrics computation
        interpolation_mode : str
            Interpolation mode: "none", "linear", or "cubic"
        dt : float
            Time step between frames
            
        Returns
        -------
        List[Tuple[float, float]]
            List of (x, y) centroid positions
        """
        # Extract raw trajectory
        raw_trajectory = []
        frame_times = []
        for idx in sorted(results_by_index.keys()):
            result = results_by_index[idx]
            if result.bev_centroid is not None:
                raw_trajectory.append(result.bev_centroid)
                frame_times.append(float(idx) * dt)
        
        if not apply_interpolation or interpolation_mode == "none" or len(raw_trajectory) < 3:
            return raw_trajectory
        
        # Apply interpolation to match metrics computation
        from scipy import interpolate as scipy_interp
        
        t = np.array(frame_times)
        data = np.array(raw_trajectory)
        t_interp = np.linspace(t[0], t[-1], len(t) * 3)
        
        if interpolation_mode == "linear":
            interp_func = scipy_interp.interp1d(
                t, data, kind='linear', axis=0,
                bounds_error=False, fill_value=(data[0], data[-1])
            )
            data_interp = interp_func(t_interp)
        elif interpolation_mode == "cubic" and len(t) >= 4:
            interp_func = scipy_interp.CubicSpline(
                t, data, axis=0, bc_type='natural', extrapolate=False
            )
            data_interp = interp_func(t_interp)
            # Handle NaN from extrapolate=False
            data_interp = np.where(
                np.isnan(data_interp),
                np.where(t_interp[:, None] < t[0], data[0], data[-1]),
                data_interp
            )
        else:
            return raw_trajectory
        
        return [tuple(pt) for pt in data_interp]


class ErrorAnalysisPlotter:
    """Generate error analysis plots for presentation."""
    
    COLORS = TrajectoryVisualizer.COLORS
    
    @staticmethod
    def plot_error_comparison(
        csv_data: List[Dict],
        output_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Create simplified error analysis plot (averaged across modes).

        Parameters
        ----------
        csv_data : List[Dict]
            Parsed CSV data with error metrics
        output_path : Optional[Path]
            If provided, save figure to this path

        Returns
        -------
        plt.Figure
            The generated figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
        fig.patch.set_facecolor('white')

        # Organize data by mode
        modes = ['none', 'linear', 'cubic']

        data_by_mode = {mode: [] for mode in modes}
        for row in csv_data:
            mode = row.get('interpolation_mode')
            if mode in modes and row.get('fraction') != 'episode_total' and row.get('fraction') not in ['averages', 'mae']:
                try:
                    data_by_mode[mode].append({
                        'fraction': float(row.get('fraction', 0)),
                        'speed_error': float(row.get('speed_error', 0)),
                        'accel_mag_error': float(row.get('accel_mag_error', 0)),
                    })
                except (ValueError, TypeError):
                    continue

        # Compute averaged errors across modes
        all_fractions = sorted(set(
            d['fraction'] for mode_data in data_by_mode.values()
            for d in mode_data
        ))

        avg_speed_errors = []
        avg_accel_errors = []

        for frac in all_fractions:
            speed_vals = []
            accel_vals = []
            for mode in modes:
                mode_data = [d for d in data_by_mode[mode] if abs(d['fraction'] - frac) < 0.01]
                if mode_data:
                    speed_vals.append(abs(mode_data[0]['speed_error']))
                    accel_vals.append(abs(mode_data[0]['accel_mag_error']))

            if speed_vals:
                avg_speed_errors.append(np.mean(speed_vals))
                avg_accel_errors.append(np.mean(accel_vals))

        # Plot 1: Speed Error
        ax1 = axes[0]
        ax1.set_facecolor('#F5F5F5')
        if avg_speed_errors:
            ax1.plot(all_fractions, avg_speed_errors, 'o-',
                    linewidth=3, markersize=8,
                    color='#1976D2', markerfacecolor='#1976D2',
                    markeredgecolor='white', markeredgewidth=2)
        ax1.set_xlabel('Lane Fraction', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Speed Error (m/s)', fontsize=12, fontweight='bold')
        ax1.set_title('Average Speed Error Along Lane', fontsize=14, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, color='#BDBDBD', linestyle='-', linewidth=0.8)

        # Plot 2: Acceleration Error
        ax2 = axes[1]
        ax2.set_facecolor('#F5F5F5')
        if avg_accel_errors:
            ax2.plot(all_fractions, avg_accel_errors, 's-',
                    linewidth=3, markersize=8,
                    color='#F57C00', markerfacecolor='#F57C00',
                    markeredgecolor='white', markeredgewidth=2)
        ax2.set_xlabel('Lane Fraction', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accel Error (m/s²)', fontsize=12, fontweight='bold')
        ax2.set_title('Average Acceleration Error Along Lane', fontsize=14, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, color='#BDBDBD', linestyle='-', linewidth=0.8)

        fig.suptitle('Performance Metrics (Averaged Across Interpolation Modes)',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Saved error analysis plot: {output_path}")

        return fig
    
    @staticmethod
    def plot_summary_metrics(
        csv_data: List[Dict],
        output_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Create simplified summary metrics bar chart (averaged across modes).

        Parameters
        ----------
        csv_data : List[Dict]
            Parsed CSV data including summary statistics
        output_path : Optional[Path]
            If provided, save figure to this path

        Returns
        -------
        plt.Figure
            The generated figure
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#F5F5F5')

        modes = ['none', 'linear', 'cubic']

        # Extract summary statistics and compute averages
        summary_data = {mode: {} for mode in modes}
        for row in csv_data:
            if row.get('test_name') == 'summary_statistics':
                mode = row.get('interpolation_mode')
                stat_type = row.get('fraction')  # 'averages' or 'mae'
                if mode in modes and stat_type == 'mae':
                    try:
                        summary_data[mode] = {
                            'speed_error': abs(float(row.get('pred_speed', 0))),
                            'accel_mag_error': abs(float(row.get('pred_accel_mag', 0))),
                            'total_break_error': abs(float(row.get('pred_total_break', 0))),
                            'end_speed_error': abs(float(row.get('pred_end_speed', 0))),
                        }
                    except (ValueError, TypeError):
                        continue

        # Compute average across modes
        metric_keys = ['speed_error', 'accel_mag_error', 'total_break_error', 'end_speed_error']
        metric_labels = ['Speed\n(m/s)', 'Acceleration\n(m/s²)', 'Break\n(m)', 'End Speed\n(m/s)']
        avg_values = []

        for key in metric_keys:
            values = [summary_data[mode].get(key, 0) for mode in modes if mode in summary_data and key in summary_data[mode]]
            if values:
                avg_values.append(np.mean(values))
            else:
                avg_values.append(0)

        # Create bar chart
        colors = ['#1976D2', '#F57C00', '#388E3C', '#7B1FA2']
        x = np.arange(len(metric_labels))
        bars = ax.bar(x, avg_values, color=colors, alpha=0.8, edgecolor='white', linewidth=2)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_xlabel('Metric Type', fontsize=13, fontweight='bold')
        ax.set_ylabel('Mean Absolute Error', fontsize=13, fontweight='bold')
        ax.set_title('Average Performance Metrics Across All Modes', fontsize=15, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y', color='#BDBDBD', linestyle='-', linewidth=0.8)

        plt.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Saved summary metrics plot: {output_path}")

        return fig


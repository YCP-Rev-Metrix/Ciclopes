"""
Trajectory visualization for BEV lane analysis.

Generates publication-quality plots showing:
- Lane boundaries in BEV space
- Ball trajectory with different interpolation modes
- Error overlays and metrics
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2

from .postprocessor import ProcessResult


class TrajectoryVisualizer:
    """Visualization tool for trajectory analysis in BEV space."""
    
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
        
        # Velocity errors
        if 'vel_x_error' in errors:
            lines.append(f"Vel X Error: {errors['vel_x_error']:>8.2f}")
        if 'vel_y_error' in errors:
            lines.append(f"Vel Y Error: {errors['vel_y_error']:>8.2f}")
        if 'speed_error' in errors:
            lines.append(f"Speed Error: {errors['speed_error']:>8.2f}")
        
        # Acceleration errors
        if 'acc_x_error' in errors:
            lines.append(f"Acc X Error: {errors['acc_x_error']:>8.2f}")
        if 'acc_y_error' in errors:
            lines.append(f"Acc Y Error: {errors['acc_y_error']:>8.2f}")
        if 'accel_mag_error' in errors:
            lines.append(f"Acc Mag Err: {errors['accel_mag_error']:>8.2f}")
        
        # Position errors
        if 'total_break_error' in errors:
            lines.append(f"Break Error:  {errors['total_break_error']:>7.3f}")
        if 'end_speed_error' in errors:
            lines.append(f"End Spd Err: {errors['end_speed_error']:>8.2f}")
        
        # Detection quality
        if 'seg_map_50_95' in errors:
            lines.append("─" * 25)
            lines.append(f"mAP@50-95:   {errors['seg_map_50_95']:>8.2%}")
        
        return '\n'.join(lines)
    
    def extract_trajectory_from_results(
        self, 
        results_by_index: Dict[int, ProcessResult]
    ) -> List[Tuple[float, float]]:
        """
        Extract trajectory points from PostProcessor results.
        
        Parameters
        ----------
        results_by_index : Dict[int, ProcessResult]
            Results dictionary from PostProcessor.process_run()
            
        Returns
        -------
        List[Tuple[float, float]]
            List of (x, y) centroid positions
        """
        trajectory = []
        for idx in sorted(results_by_index.keys()):
            result = results_by_index[idx]
            if result.bev_centroid is not None:
                trajectory.append(result.bev_centroid)
        return trajectory


class ErrorAnalysisPlotter:
    """Generate error analysis plots for presentation."""
    
    COLORS = TrajectoryVisualizer.COLORS
    
    @staticmethod
    def plot_error_comparison(
        csv_data: List[Dict],
        output_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Create comprehensive error analysis plots.
        
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
        fig = plt.figure(figsize=(16, 10), dpi=150)
        fig.patch.set_facecolor('white')
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # Organize data by mode
        modes = ['none', 'linear', 'cubic']
        mode_colors = {
            'none': ErrorAnalysisPlotter.COLORS['trajectory_none'],
            'linear': ErrorAnalysisPlotter.COLORS['trajectory_linear'],
            'cubic': ErrorAnalysisPlotter.COLORS['trajectory_cubic'],
        }
        
        data_by_mode = {mode: [] for mode in modes}
        for row in csv_data:
            mode = row.get('interpolation_mode')
            if mode in modes and row.get('fraction') != 'episode_total':
                try:
                    data_by_mode[mode].append({
                        'fraction': float(row.get('fraction', 0)),
                        'speed_error': float(row.get('speed_error', 0)),
                        'accel_mag_error': float(row.get('accel_mag_error', 0)),
                        'vel_x_error': float(row.get('vel_x_error', 0)),
                        'vel_y_error': float(row.get('vel_y_error', 0)),
                        'acc_x_error': float(row.get('acc_x_error', 0)),
                        'acc_y_error': float(row.get('acc_y_error', 0)),
                    })
                except (ValueError, TypeError):
                    continue
        
        # 1. Speed Error vs Lane Fraction
        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_facecolor('#FAFAFA')
        for mode in modes:
            if data_by_mode[mode]:
                fractions = [d['fraction'] for d in data_by_mode[mode]]
                speed_errors = [abs(d['speed_error']) for d in data_by_mode[mode]]
                ax1.plot(fractions, speed_errors, 'o-', linewidth=2, markersize=6,
                        label=mode.capitalize(), color=mode_colors[mode], alpha=0.8)
        ax1.set_xlabel('Lane Fraction', fontsize=11, fontweight='bold')
        ax1.set_ylabel('|Speed Error| (units/s)', fontsize=11, fontweight='bold')
        ax1.set_title('Speed Error Along Lane', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, color=ErrorAnalysisPlotter.COLORS['grid'])
        
        # 2. Acceleration Error vs Lane Fraction
        ax2 = fig.add_subplot(gs[1, :])
        ax2.set_facecolor('#FAFAFA')
        for mode in modes:
            if data_by_mode[mode]:
                fractions = [d['fraction'] for d in data_by_mode[mode]]
                accel_errors = [abs(d['accel_mag_error']) for d in data_by_mode[mode]]
                ax2.plot(fractions, accel_errors, 's-', linewidth=2, markersize=6,
                        label=mode.capitalize(), color=mode_colors[mode], alpha=0.8)
        ax2.set_xlabel('Lane Fraction', fontsize=11, fontweight='bold')
        ax2.set_ylabel('|Accel Error| (units/s²)', fontsize=11, fontweight='bold')
        ax2.set_title('Acceleration Error Along Lane', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, color=ErrorAnalysisPlotter.COLORS['grid'])
        
        # 3. Error Distribution - Speed
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.set_facecolor('#FAFAFA')
        speed_errors_by_mode = []
        labels = []
        for mode in modes:
            if data_by_mode[mode]:
                speed_errors_by_mode.append([abs(d['speed_error']) for d in data_by_mode[mode]])
                labels.append(mode.capitalize())
        bp1 = ax3.boxplot(speed_errors_by_mode, labels=labels, patch_artist=True)
        for patch, mode in zip(bp1['boxes'], modes):
            patch.set_facecolor(mode_colors[mode])
            patch.set_alpha(0.6)
        ax3.set_ylabel('|Speed Error|', fontsize=10, fontweight='bold')
        ax3.set_title('Speed Error Distribution', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y', color=ErrorAnalysisPlotter.COLORS['grid'])
        
        # 4. Error Distribution - Acceleration
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.set_facecolor('#FAFAFA')
        accel_errors_by_mode = []
        for mode in modes:
            if data_by_mode[mode]:
                accel_errors_by_mode.append([abs(d['accel_mag_error']) for d in data_by_mode[mode]])
        bp2 = ax4.boxplot(accel_errors_by_mode, labels=labels, patch_artist=True)
        for patch, mode in zip(bp2['boxes'], modes):
            patch.set_facecolor(mode_colors[mode])
            patch.set_alpha(0.6)
        ax4.set_ylabel('|Accel Error|', fontsize=10, fontweight='bold')
        ax4.set_title('Accel Error Distribution', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y', color=ErrorAnalysisPlotter.COLORS['grid'])
        
        # 5. Error Correlation (Vel X vs Vel Y)
        ax5 = fig.add_subplot(gs[2, 2])
        ax5.set_facecolor('#FAFAFA')
        for mode in modes:
            if data_by_mode[mode]:
                vel_x_errors = [d['vel_x_error'] for d in data_by_mode[mode]]
                vel_y_errors = [d['vel_y_error'] for d in data_by_mode[mode]]
                ax5.scatter(vel_x_errors, vel_y_errors, s=50, alpha=0.6,
                           label=mode.capitalize(), color=mode_colors[mode])
        ax5.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.3)
        ax5.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.3)
        ax5.set_xlabel('Vel X Error', fontsize=10, fontweight='bold')
        ax5.set_ylabel('Vel Y Error', fontsize=10, fontweight='bold')
        ax5.set_title('Velocity Error Correlation', fontsize=11, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3, color=ErrorAnalysisPlotter.COLORS['grid'])
        
        fig.suptitle('End-to-End Performance Analysis', 
                    fontsize=16, fontweight='bold', y=0.995)
        
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
        Create summary metrics comparison bar chart.
        
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
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
        fig.patch.set_facecolor('white')
        
        modes = ['none', 'linear', 'cubic']
        mode_colors = {
            'none': ErrorAnalysisPlotter.COLORS['trajectory_none'],
            'linear': ErrorAnalysisPlotter.COLORS['trajectory_linear'],
            'cubic': ErrorAnalysisPlotter.COLORS['trajectory_cubic'],
        }
        
        # Extract summary statistics
        summary_data = {mode: {} for mode in modes}
        for row in csv_data:
            if row.get('test_name') == 'summary_statistics':
                mode = row.get('interpolation_mode')
                stat_type = row.get('fraction')  # 'averages' or 'mae'
                if mode in modes and stat_type in ['averages', 'mae']:
                    if stat_type not in summary_data[mode]:
                        summary_data[mode][stat_type] = {}
                    try:
                        summary_data[mode][stat_type].update({
                            'vel_x_error': abs(float(row.get('vel_x_error', 0))),
                            'vel_y_error': abs(float(row.get('vel_y_error', 0))),
                            'speed_error': abs(float(row.get('speed_error', 0))),
                            'acc_x_error': abs(float(row.get('acc_x_error', 0))),
                            'acc_y_error': abs(float(row.get('acc_y_error', 0))),
                            'accel_mag_error': abs(float(row.get('accel_mag_error', 0))),
                            'total_break_error': abs(float(row.get('total_break_error', 0))),
                            'end_speed_error': abs(float(row.get('end_speed_error', 0))),
                        })
                    except (ValueError, TypeError):
                        continue
        
        # Plot MAE metrics
        ax1 = axes[0, 0]
        ax1.set_facecolor('#FAFAFA')
        metrics = ['vel_x_error', 'vel_y_error', 'speed_error']
        metric_labels = ['Vel X', 'Vel Y', 'Speed']
        x = np.arange(len(metric_labels))
        width = 0.25
        
        for i, mode in enumerate(modes):
            if 'mae' in summary_data[mode]:
                values = [summary_data[mode]['mae'].get(m, 0) for m in metrics]
                ax1.bar(x + i * width, values, width, label=mode.capitalize(),
                       color=mode_colors[mode], alpha=0.8)
        
        ax1.set_xlabel('Metric', fontsize=11, fontweight='bold')
        ax1.set_ylabel('MAE', fontsize=11, fontweight='bold')
        ax1.set_title('Velocity Error (MAE)', fontsize=12, fontweight='bold')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(metric_labels)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y', color=ErrorAnalysisPlotter.COLORS['grid'])
        
        # Plot acceleration MAE
        ax2 = axes[0, 1]
        ax2.set_facecolor('#FAFAFA')
        metrics = ['acc_x_error', 'acc_y_error', 'accel_mag_error']
        metric_labels = ['Acc X', 'Acc Y', 'Acc Mag']
        x = np.arange(len(metric_labels))
        
        for i, mode in enumerate(modes):
            if 'mae' in summary_data[mode]:
                values = [summary_data[mode]['mae'].get(m, 0) for m in metrics]
                ax2.bar(x + i * width, values, width, label=mode.capitalize(),
                       color=mode_colors[mode], alpha=0.8)
        
        ax2.set_xlabel('Metric', fontsize=11, fontweight='bold')
        ax2.set_ylabel('MAE', fontsize=11, fontweight='bold')
        ax2.set_title('Acceleration Error (MAE)', fontsize=12, fontweight='bold')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(metric_labels)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y', color=ErrorAnalysisPlotter.COLORS['grid'])
        
        # Plot position errors
        ax3 = axes[1, 0]
        ax3.set_facecolor('#FAFAFA')
        metrics = ['total_break_error', 'end_speed_error']
        metric_labels = ['Break Error', 'End Speed']
        x = np.arange(len(metric_labels))
        
        for i, mode in enumerate(modes):
            if 'mae' in summary_data[mode]:
                values = [summary_data[mode]['mae'].get(m, 0) for m in metrics]
                ax3.bar(x + i * width, values, width, label=mode.capitalize(),
                       color=mode_colors[mode], alpha=0.8)
        
        ax3.set_xlabel('Metric', fontsize=11, fontweight='bold')
        ax3.set_ylabel('MAE', fontsize=11, fontweight='bold')
        ax3.set_title('Position & End Speed Error (MAE)', fontsize=12, fontweight='bold')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(metric_labels)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y', color=ErrorAnalysisPlotter.COLORS['grid'])
        
        # Overall performance comparison (normalized)
        ax4 = axes[1, 1]
        ax4.set_facecolor('#FAFAFA')
        
        # Calculate overall error score (lower is better)
        overall_scores = []
        for mode in modes:
            if 'mae' in summary_data[mode]:
                score = (
                    summary_data[mode]['mae'].get('speed_error', 0) +
                    summary_data[mode]['mae'].get('accel_mag_error', 0) / 10 +  # Scale down accel
                    summary_data[mode]['mae'].get('total_break_error', 0) * 10  # Scale up break
                )
                overall_scores.append(score)
            else:
                overall_scores.append(0)
        
        x = np.arange(len(modes))
        bars = ax4.bar(x, overall_scores, color=[mode_colors[m] for m in modes], alpha=0.8)
        ax4.set_xlabel('Interpolation Mode', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Composite Error Score', fontsize=11, fontweight='bold')
        ax4.set_title('Overall Performance (Lower is Better)', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels([m.capitalize() for m in modes])
        ax4.grid(True, alpha=0.3, axis='y', color=ErrorAnalysisPlotter.COLORS['grid'])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        fig.suptitle('Summary Performance Metrics', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Saved summary metrics plot: {output_path}")
        
        return fig


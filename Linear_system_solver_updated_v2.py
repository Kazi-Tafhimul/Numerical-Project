


import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton, 
                           QComboBox, QTextEdit, QTabWidget, QSpinBox, QDoubleSpinBox,
                           QSlider, QFrame, QSplitter, QScrollArea, QGroupBox,
                           QMessageBox, QFileDialog, QProgressBar, QCheckBox)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation, QRect
from PyQt6.QtGui import QPainter, QColor, QFont, QPen, QBrush, QLinearGradient
import json
from datetime import datetime
import time

class MatrixWidget(QWidget):
   
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.matrix = None
        self.highlighted_cells = []
        self.animation_step = 0
        self.cell_colors = {}
        self.operation_text = ""
        self.setMinimumSize(400, 300)
        
    def set_matrix(self, matrix, highlighted_cells=None, operation_text=""):
        """Set matrix data and highlight specific cells"""
        self.matrix = matrix.copy() if matrix is not None else None
        self.highlighted_cells = highlighted_cells or []
        self.operation_text = operation_text
        self.update()
        
    def animate_operation(self, from_cells, to_cells, operation_type="elimination"):
        """Animate matrix operations using custom animation engine"""
        self.animation_step = 0
        self.from_cells = from_cells
        self.to_cells = to_cells
        self.operation_type = operation_type
        
        # Start animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(100)  # Update every 100ms
        
    def update_animation(self):
        """Update animation frame"""
        self.animation_step += 1
        if self.animation_step > 10:  # Animation duration
            self.animation_timer.stop()
            self.animation_step = 0
        self.update()
        
    def paintEvent(self, event):
       
        if self.matrix is None:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Calculate cell dimensions
        rows, cols = self.matrix.shape
        cell_width = (self.width() - 40) / cols
        cell_height = (self.height() - 80) / rows
        
        # Draw matrix cells
        for i in range(rows):
            for j in range(cols):
                x = 20 + j * cell_width
                y = 20 + i * cell_height
                
                # Determine cell color based on highlighting and animation
                color = QColor(255, 255, 255)  # Default white
                
                if (i, j) in self.highlighted_cells:
                    color = QColor(255, 200, 200)  # Light red for highlighted
                    
                if hasattr(self, 'animation_step') and self.animation_step > 0:
                    if (i, j) in getattr(self, 'from_cells', []):
                        intensity = 255 - (self.animation_step * 20)
                        color = QColor(255, intensity, intensity)
                    elif (i, j) in getattr(self, 'to_cells', []):
                        intensity = 200 + (self.animation_step * 5)
                        color = QColor(intensity, 255, intensity)
                
                # Draw cell background
                painter.fillRect(int(x), int(y), int(cell_width), int(cell_height), color)
                
                # Draw cell border
                painter.setPen(QPen(QColor(0, 0, 0), 1))
                painter.drawRect(int(x), int(y), int(cell_width), int(cell_height))
                
                # Draw cell value
                painter.setPen(QPen(QColor(0, 0, 0), 2))
                font = QFont("Arial", 12, QFont.Weight.Bold)
                painter.setFont(font)
                value_text = f"{self.matrix[i, j]:.2f}"
                text_rect = QRect(int(x), int(y), int(cell_width), int(cell_height))
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, value_text)
        
        # Draw operation text
        if self.operation_text:
            painter.setPen(QPen(QColor(0, 0, 255), 2))
            font = QFont("Arial", 14, QFont.Weight.Bold)
            painter.setFont(font)
            text_rect = QRect(0, self.height() - 40, self.width(), 30)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self.operation_text)

class ConvergenceWidget(QWidget):
    
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.errors = []
        self.solutions = []
        self.method = ""
        self.residuals = []
        self.steps_data = []
        self.setMinimumSize(400, 300)
        
    def set_data(self, errors=None, solutions=None, method="", residuals=None, steps_data=None):
        """Set convergence/analysis data"""
        self.errors = errors or []
        self.solutions = solutions or []
        self.method = method
        self.residuals = residuals or []
        self.steps_data = steps_data or []
        self.update()
        
    def paintEvent(self, event):
        """Custom paint event for convergence plots and method analysis"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw background
        painter.fillRect(self.rect(), QColor(245, 245, 245))
        
        # Calculate plot area
        margin = 50
        plot_width = self.width() - 2 * margin
        plot_height = self.height() - 2 * margin
        
        if self.method in ["Jacobi", "Gauss-Seidel"] and self.errors:
            self.draw_convergence_plot(painter, margin, plot_width, plot_height)
        elif self.method == "Gaussian Elimination" and self.steps_data:
            self.draw_elimination_progress(painter, margin, plot_width, plot_height)
        elif self.method == "Matrix Inversion":
            self.draw_inversion_info(painter, margin, plot_width, plot_height)
        else:
            self.draw_no_data_message(painter)
    
    def draw_convergence_plot(self, painter, margin, plot_width, plot_height):
        """Draw convergence plot for iterative methods"""
        # Draw axes
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        painter.drawLine(margin, self.height() - margin, 
                        self.width() - margin, self.height() - margin)  # X-axis
        painter.drawLine(margin, margin, margin, self.height() - margin)  # Y-axis
        
        # Draw error curve
        if len(self.errors) > 1:
            painter.setPen(QPen(QColor(255, 0, 0), 3))
            
            # Use logarithmic scale for better visualization
            log_errors = [np.log10(max(e, 1e-16)) for e in self.errors]
            max_log_error = max(log_errors)
            min_log_error = min(log_errors)
            error_range = max_log_error - min_log_error if max_log_error > min_log_error else 1
            
            for i in range(len(log_errors) - 1):
                x1 = margin + (i * plot_width) / (len(log_errors) - 1)
                y1 = self.height() - margin - ((log_errors[i] - min_log_error) / error_range) * plot_height
                x2 = margin + ((i + 1) * plot_width) / (len(log_errors) - 1)
                y2 = self.height() - margin - ((log_errors[i + 1] - min_log_error) / error_range) * plot_height
                
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        
        # Draw labels and title
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        font = QFont("Arial", 12, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(10, 25, f"{self.method} Convergence")
        
        font = QFont("Arial", 10)
        painter.setFont(font)
        painter.drawText(10, 45, "Log₁₀(Error)")
        painter.drawText(self.width() - 80, self.height() - 10, "Iteration")
        
        # Show final error
        if self.errors:
            painter.drawText(10, self.height() - 10, f"Final Error: {self.errors[-1]:.2e}")
    
    def draw_elimination_progress(self, painter, margin, plot_width, plot_height):
        """Draw progress visualization for Gaussian Elimination"""
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        font = QFont("Arial", 14, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(margin, 40, "Gaussian Elimination Progress")
        
        # Draw progress bars for elimination steps
        if self.steps_data:
            font = QFont("Arial", 12)
            painter.setFont(font)
            
            forward_steps = len([s for s in self.steps_data if 'Eliminate' in s.get('step', '') or 'Swap' in s.get('step', '')])
            back_steps = len([s for s in self.steps_data if 'Back substitution' in s.get('step', '')])
            
            # Forward elimination progress
            painter.setPen(QPen(QColor(0, 100, 200), 2))
            painter.drawText(margin, 80, f"Forward Elimination: {forward_steps} operations")
            
            if forward_steps > 0:
                rect_width = min(plot_width * 0.8, 300)
                painter.fillRect(margin, 90, int(rect_width), 20, QColor(100, 150, 255))
            
            # Back substitution progress
            painter.drawText(margin, 130, f"Back Substitution: {back_steps} steps")
            
            if back_steps > 0:
                rect_width = min(plot_width * 0.8, 300)
                painter.fillRect(margin, 140, int(rect_width), 20, QColor(150, 255, 150))
            
            # Show pivot operations
            swap_steps = len([s for s in self.steps_data if 'Swap' in s.get('step', '')])
            painter.drawText(margin, 180, f"Row Swaps: {swap_steps}")
            
            # Show elimination operations
            elim_steps = len([s for s in self.steps_data if 'Eliminate' in s.get('step', '')])
            painter.drawText(margin, 200, f"Elimination Operations: {elim_steps}")
    
    def draw_inversion_info(self, painter, margin, plot_width, plot_height):
        """Draw information for Matrix Inversion method"""
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        font = QFont("Arial", 14, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(margin, 40, "Matrix Inversion Analysis")
        
        font = QFont("Arial", 12)
        painter.setFont(font)
        
        # Method description
        painter.drawText(margin, 80, "Method: Direct solution using A⁻¹ × b")
        painter.drawText(margin, 100, "• Computes matrix inverse using LU decomposition")
        painter.drawText(margin, 120, "• Multiplies inverse by vector b")
        painter.drawText(margin, 140, "• Single-step solution (no iterations)")
        
        # Computational complexity
        painter.setPen(QPen(QColor(150, 0, 0), 1))
        painter.drawText(margin, 180, "Computational Complexity:")
        painter.drawText(margin, 200, "• Time: O(n³) for inversion + O(n²) for multiplication")
        painter.drawText(margin, 220, "• Space: O(n²) for storing inverse matrix")
        
        # Advantages/Disadvantages
        painter.setPen(QPen(QColor(0, 120, 0), 1))
        painter.drawText(margin, 260, "✓ Exact solution (within numerical precision)")
        painter.drawText(margin, 280, "✓ Good for multiple right-hand sides")
        
        painter.setPen(QPen(QColor(200, 100, 0), 1))
        painter.drawText(margin, 300, "⚠ Sensitive to matrix conditioning")
        painter.drawText(margin, 320, "⚠ High memory usage for large matrices")
    
    def draw_no_data_message(self, painter):
        """Draw message when no data is available"""
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        font = QFont("Arial", 14)
        painter.setFont(font)
        
        text = "No convergence data available"
        if self.method:
            if self.method in ["Jacobi", "Gauss-Seidel"]:
                text = f"Waiting for {self.method} iterations..."
            else:
                text = f"{self.method}: Direct method (no convergence plot)"
        
        text_rect = painter.fontMetrics().boundingRect(text)
        x = (self.width() - text_rect.width()) // 2
        y = (self.height() - text_rect.height()) // 2
        painter.drawText(x, y, text)

class SolutionThread(QThread):
   
    
    progress_updated = pyqtSignal(int)
    step_completed = pyqtSignal(dict)
    solution_completed = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, matrix_A, vector_b, method, tolerance=1e-6, max_iterations=100):
        super().__init__()
        self.matrix_A = matrix_A
        self.vector_b = vector_b
        self.method = method
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.should_stop = False
        
    def run(self):
        """Run the solution method in separate thread"""
        try:
            if self.method == "Gaussian Elimination":
                solution = self.gaussian_elimination()
            elif self.method == "Matrix Inversion":
                solution = self.matrix_inversion()
            elif self.method == "Jacobi":
                solution = self.jacobi_method()
            elif self.method == "Gauss-Seidel":
                solution = self.gauss_seidel_method()
            else:
                raise ValueError(f"Unknown method: {self.method}")
                
            self.solution_completed.emit(solution)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def gaussian_elimination(self):
        
        A = self.matrix_A.copy()
        b = self.vector_b.copy()
        n = len(b)
        
        # Forward elimination
        for i in range(n):
            if self.should_stop:
                break
                
            # Find pivot
            max_row = i
            for k in range(i+1, n):
                if abs(A[k][i]) > abs(A[max_row][i]):
                    max_row = k
            
            # Swap rows if needed
            if max_row != i:
                A[[i, max_row]] = A[[max_row, i]]
                b[[i, max_row]] = b[[max_row, i]]
                
                step_data = {
                    'step': f"Swap rows {i+1} and {max_row+1}",
                    'matrix': A.copy(),
                    'vector': b.copy(),
                    'highlighted_cells': [(i, j) for j in range(n)] + [(max_row, j) for j in range(n)],
                    'operation': f"R{i+1} ↔ R{max_row+1}"
                }
                self.step_completed.emit(step_data)
                time.sleep(0.1)  # Small delay for animation
            
            # Eliminate column
            for k in range(i+1, n):
                if self.should_stop:
                    break
                    
                if A[i][i] != 0:
                    factor = A[k][i] / A[i][i]
                    for j in range(i, n):
                        A[k][j] -= factor * A[i][j]
                    b[k] -= factor * b[i]
                    
                    step_data = {
                        'step': f"Eliminate A[{k+1},{i+1}]",
                        'matrix': A.copy(),
                        'vector': b.copy(),
                        'highlighted_cells': [(k, j) for j in range(n)] + [(i, j) for j in range(n)],
                        'operation': f"R{k+1} = R{k+1} - {factor:.3f} × R{i+1}"
                    }
                    self.step_completed.emit(step_data)
                    time.sleep(0.1)
            
            progress = int(((i + 1) / n) * 50)  # 50% for forward elimination
            self.progress_updated.emit(progress)
        
        # Back substitution
        x = np.zeros(n)
        for i in range(n-1, -1, -1):
            if self.should_stop:
                break
                
            x[i] = b[i]
            for j in range(i+1, n):
                x[i] -= A[i][j] * x[j]
            x[i] /= A[i][i]
            
            step_data = {
                'step': f"Back substitution: x[{i+1}] = {x[i]:.6f}",
                'matrix': A.copy(),
                'vector': b.copy(),
                'solution': x.copy(),
                'highlighted_cells': [(i, j) for j in range(n)],
                'operation': f"x[{i+1}] = {x[i]:.6f}"
            }
            self.step_completed.emit(step_data)
            time.sleep(0.1)
            
            progress = 50 + int(((n - i) / n) * 50)  # 50% for back substitution
            self.progress_updated.emit(progress)
        
        return x
    
    def matrix_inversion(self):
      
        try:
            A_inv = np.linalg.inv(self.matrix_A)
            solution = A_inv @ self.vector_b
            
            step_data = {
                'step': 'Matrix inversion completed',
                'matrix': A_inv,
                'vector': self.vector_b,
                'solution': solution,
                'operation': 'x = A⁻¹ × b'
            }
            self.step_completed.emit(step_data)
            self.progress_updated.emit(100)
            
            return solution
        except np.linalg.LinAlgError:
            raise ValueError("Matrix is singular and cannot be inverted")
    
    def jacobi_method(self):
        
        A = self.matrix_A
        b = self.vector_b
        n = len(b)
        x = np.zeros(n)
        
        for iteration in range(self.max_iterations):
            if self.should_stop:
                break
                
            x_new = np.zeros(n)
            
            for i in range(n):
                if A[i][i] == 0:
                    raise ValueError(f"Zero diagonal element at position ({i},{i})")
                    
                sum_ax = sum(A[i][j] * x[j] for j in range(n) if j != i)
                x_new[i] = (b[i] - sum_ax) / A[i][i]
            
            error = np.linalg.norm(x_new - x)
            
            step_data = {
                'step': f"Jacobi iteration {iteration+1}",
                'matrix': A,
                'vector': b,
                'solution': x_new.copy(),
                'error': error,
                'operation': f"Iteration {iteration+1}, Error: {error:.6f}"
            }
            self.step_completed.emit(step_data)
            
            if error < self.tolerance:
                break
                
            x = x_new
            progress = min(int((iteration / self.max_iterations) * 100), 100)
            self.progress_updated.emit(progress)
            time.sleep(0.05)
        
        return x
    
    def gauss_seidel_method(self):

        A = self.matrix_A
        b = self.vector_b
        n = len(b)
        x = np.zeros(n)
        
        for iteration in range(self.max_iterations):
            if self.should_stop:
                break
                
            x_old = x.copy()
            
            for i in range(n):
                if A[i][i] == 0:
                    raise ValueError(f"Zero diagonal element at position ({i},{i})")
                    
                sum_ax = sum(A[i][j] * x[j] for j in range(n) if j != i)
                x[i] = (b[i] - sum_ax) / A[i][i]
            
            error = np.linalg.norm(x - x_old)
            
            step_data = {
                'step': f"Gauss-Seidel iteration {iteration+1}",
                'matrix': A,
                'vector': b,
                'solution': x.copy(),
                'error': error,
                'operation': f"Iteration {iteration+1}, Error: {error:.6f}"
            }
            self.step_completed.emit(step_data)
            
            if error < self.tolerance:
                break
                
            progress = min(int((iteration / self.max_iterations) * 100), 100)
            self.progress_updated.emit(progress)
            time.sleep(0.05)
        
        return x
    
    def stop(self):
        """Stop the solution process"""
        self.should_stop = True

class LinearSystemSolver(QMainWindow):

    
    def __init__(self):
        super().__init__()
        self.matrix_A = None
        self.vector_b = None
        self.solution = None
        self.iteration_data = []
        self.current_method = None
        self.solver_thread = None
        
        self.init_ui()
        self.generate_random_system()  # Start with a random system
        
    def init_ui(self):
       
        self.setWindowTitle("Linear System Solver - PyQt6 Enhanced")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Input and controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Results and visualization
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([400, 1000])
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
    
    def create_left_panel(self):
        """Create the left control panel"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Matrix input section
        input_group = QGroupBox("System Input")
        input_layout = QVBoxLayout(input_group)
        
        # Matrix size controls
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Matrix Size:"))
        self.size_spinbox = QSpinBox()
        self.size_spinbox.setRange(2, 6)
        self.size_spinbox.setValue(3)
        self.size_spinbox.valueChanged.connect(self.update_matrix_size)
        size_layout.addWidget(self.size_spinbox)
        
        self.random_button = QPushButton("Generate Random")
        self.random_button.clicked.connect(self.generate_random_system)
        size_layout.addWidget(self.random_button)
        
        input_layout.addLayout(size_layout)
        
        # Matrix input area
        self.matrix_scroll = QScrollArea()
        self.matrix_widget = QWidget()
        self.matrix_layout = QGridLayout(self.matrix_widget)
        self.matrix_scroll.setWidget(self.matrix_widget)
        self.matrix_scroll.setWidgetResizable(True)
        self.matrix_scroll.setMaximumHeight(300)
        input_layout.addWidget(self.matrix_scroll)
        
        left_layout.addWidget(input_group)
        
        # Solution method section
        method_group = QGroupBox("Solution Method")
        method_layout = QVBoxLayout(method_group)
        
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Gaussian Elimination", "Matrix Inversion", "Jacobi", "Gauss-Seidel"])
        method_layout.addWidget(self.method_combo)
        
        # Iterative method parameters
        params_layout = QGridLayout()
        params_layout.addWidget(QLabel("Tolerance:"), 0, 0)
        self.tolerance_spinbox = QDoubleSpinBox()
        self.tolerance_spinbox.setDecimals(8)
        self.tolerance_spinbox.setRange(1e-10, 1e-2)
        self.tolerance_spinbox.setValue(1e-6)
        self.tolerance_spinbox.setSingleStep(1e-6)
        params_layout.addWidget(self.tolerance_spinbox, 0, 1)
        
        params_layout.addWidget(QLabel("Max Iterations:"), 1, 0)
        self.max_iter_spinbox = QSpinBox()
        self.max_iter_spinbox.setRange(10, 1000)
        self.max_iter_spinbox.setValue(100)
        params_layout.addWidget(self.max_iter_spinbox, 1, 1)
        
        method_layout.addLayout(params_layout)
        
        # Animation controls
        anim_layout = QHBoxLayout()
        anim_layout.addWidget(QLabel("Animation:"))
        self.animation_checkbox = QCheckBox("Enable")
        self.animation_checkbox.setChecked(True)
        anim_layout.addWidget(self.animation_checkbox)
        method_layout.addLayout(anim_layout)
        
        left_layout.addWidget(method_group)
        
        # Control buttons
        button_layout = QVBoxLayout()
        
        self.solve_button = QPushButton("Solve System")
        self.solve_button.clicked.connect(self.solve_system)
        self.solve_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        button_layout.addWidget(self.solve_button)
        
        self.stop_button = QPushButton("Stop Solution")
        self.stop_button.clicked.connect(self.stop_solution)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; }")
        button_layout.addWidget(self.stop_button)
        
        left_layout.addLayout(button_layout)
        
        # File operations
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout(file_group)
        
        save_button = QPushButton("Save System")
        save_button.clicked.connect(self.save_system)
        file_layout.addWidget(save_button)
        
        load_button = QPushButton("Load System")
        load_button.clicked.connect(self.load_system)
        file_layout.addWidget(load_button)
        
        export_button = QPushButton("Export Results")
        export_button.clicked.connect(self.export_results)
        file_layout.addWidget(export_button)
        
        left_layout.addWidget(file_group)
        
        left_layout.addStretch()
        
        return left_widget
    
    def create_right_panel(self):
        """Create the right results panel"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        right_layout.addWidget(self.tab_widget)
        
        # Matrix Animation tab
        self.matrix_tab = QWidget()
        matrix_layout = QVBoxLayout(self.matrix_tab)
        
        self.matrix_animation = MatrixWidget()
        matrix_layout.addWidget(self.matrix_animation)
        
        self.tab_widget.addTab(self.matrix_tab, "Matrix Animation")
        
        # Convergence tab
        self.convergence_tab = QWidget()
        convergence_layout = QVBoxLayout(self.convergence_tab)
        
        self.convergence_widget = ConvergenceWidget()
        convergence_layout.addWidget(self.convergence_widget)
        
        self.tab_widget.addTab(self.convergence_tab, "Convergence")
        
        # Solution tab
        self.solution_tab = QWidget()
        solution_layout = QVBoxLayout(self.solution_tab)
        
        self.solution_text = QTextEdit()
        self.solution_text.setReadOnly(True)
        self.solution_text.setFont(QFont("Courier", 10))
        solution_layout.addWidget(self.solution_text)
        
        self.tab_widget.addTab(self.solution_tab, "Solution")
        
        # Steps tab
        self.steps_tab = QWidget()
        steps_layout = QVBoxLayout(self.steps_tab)
        
        self.steps_text = QTextEdit()
        self.steps_text.setReadOnly(True)
        self.steps_text.setFont(QFont("Courier", 9))
        steps_layout.addWidget(self.steps_text)
        
        self.tab_widget.addTab(self.steps_tab, "Steps")
        
        return right_widget
    
    def update_matrix_size(self):
        """Update matrix input fields based on size"""
        size = self.size_spinbox.value()
        
        # Clear existing widgets
        for i in reversed(range(self.matrix_layout.count())):
            self.matrix_layout.itemAt(i).widget().setParent(None)
        
        # Create matrix A inputs
        self.matrix_entries = []
        for i in range(size):
            row_entries = []
            for j in range(size):
                entry = QLineEdit()
                entry.setText("0")
                entry.setMaximumWidth(80)
                self.matrix_layout.addWidget(entry, i, j)
                row_entries.append(entry)
            self.matrix_entries.append(row_entries)
        
        # Add equals sign
        for i in range(size):
            equals_label = QLabel("=")
            equals_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.matrix_layout.addWidget(equals_label, i, size)
        
        # Create vector b inputs
        self.vector_entries = []
        for i in range(size):
            entry = QLineEdit()
            entry.setText("0")
            entry.setMaximumWidth(80)
            self.matrix_layout.addWidget(entry, i, size + 1)
            self.vector_entries.append(entry)
    
    def generate_random_system(self):
       
        size = self.size_spinbox.value()
        
        # Update matrix size if needed
        if not hasattr(self, 'matrix_entries') or len(self.matrix_entries) != size:
            self.update_matrix_size()
        
        # Generate random matrix A (ensure it's well-conditioned)
        A = np.random.randint(-10, 11, (size, size))
        
        # Ensure diagonal dominance for better conditioning
        for i in range(size):
            A[i, i] = sum(abs(A[i, j]) for j in range(size) if j != i) + np.random.randint(1, 5)
        
        # Generate random vector b
        b = np.random.randint(-20, 21, size)
        
        # Update input fields
        for i in range(size):
            for j in range(size):
                self.matrix_entries[i][j].setText(str(A[i, j]))
        
        for i in range(size):
            self.vector_entries[i].setText(str(b[i]))
    
    def get_matrix_data(self):
        """Get matrix data from input fields"""
        size = self.size_spinbox.value()
        
        try:
            # Get matrix A
            A = np.zeros((size, size))
            for i in range(size):
                for j in range(size):
                    A[i, j] = float(self.matrix_entries[i][j].text())
            
            # Get vector b
            b = np.zeros(size)
            for i in range(size):
                b[i] = float(self.vector_entries[i].text())
            
            return A, b
        
        except ValueError as e:
            raise ValueError("Invalid input: Please enter numeric values only")
    
    def solve_system(self):
      
        try:
            # Get matrix data
            self.matrix_A, self.vector_b = self.get_matrix_data()
            
            # Validate matrix
            if np.linalg.det(self.matrix_A) == 0:
                QMessageBox.warning(self, "Warning", "Matrix is singular! Solution may not exist or be unique.")
            
            # Get solution parameters
            method = self.method_combo.currentText()
            tolerance = self.tolerance_spinbox.value()
            max_iterations = self.max_iter_spinbox.value()
            
            # Clear previous results
            self.clear_results()
            
            # Update UI for solving state
            self.solve_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.statusBar().showMessage(f"Solving using {method}...")
            
            # Store current method for visualization
            self.current_method = method
            self.iteration_data = []
            
            # Create and start solver thread
            self.solver_thread = SolutionThread(
                self.matrix_A, self.vector_b, method, tolerance, max_iterations
            )
            
            # Connect signals
            self.solver_thread.progress_updated.connect(self.update_progress)
            self.solver_thread.step_completed.connect(self.process_solution_step)
            self.solver_thread.solution_completed.connect(self.solution_finished)
            self.solver_thread.error_occurred.connect(self.solution_error)
            
            # Start the thread
            self.solver_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to solve system: {str(e)}")
            self.reset_ui_state()
    
    def stop_solution(self):
        """Stop the current solution process"""
        if self.solver_thread and self.solver_thread.isRunning():
            self.solver_thread.stop()
            self.solver_thread.wait(3000)  # Wait up to 3 seconds
            
        self.reset_ui_state()
        self.statusBar().showMessage("Solution stopped by user")
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def process_solution_step(self, step_data):
       
        # Store step data
        self.iteration_data.append(step_data)
        
        # Update matrix animation if enabled
        if self.animation_checkbox.isChecked():
            matrix_to_show = step_data.get('matrix', self.matrix_A)
            highlighted_cells = step_data.get('highlighted_cells', [])
            operation_text = step_data.get('operation', '')
            
            self.matrix_animation.set_matrix(matrix_to_show, highlighted_cells, operation_text)
        
        # Update steps text
        step_text = f"Step {len(self.iteration_data)}: {step_data.get('step', '')}\n"
        if 'operation' in step_data:
            step_text += f"Operation: {step_data['operation']}\n"
        if 'error' in step_data:
            step_text += f"Error: {step_data['error']:.6e}\n"
        step_text += "-" * 50 + "\n"
        
        self.steps_text.append(step_text)
        
        # Update convergence data for iterative methods
        if self.current_method in ["Jacobi", "Gauss-Seidel"] and 'error' in step_data:
            errors = [step.get('error', 0) for step in self.iteration_data if 'error' in step]
            self.convergence_widget.set_data(errors=errors, method=self.current_method)
        elif self.current_method == "Gaussian Elimination":
            self.convergence_widget.set_data(steps_data=self.iteration_data, method=self.current_method)
        elif self.current_method == "Matrix Inversion":
            self.convergence_widget.set_data(method=self.current_method)
    
    def solution_finished(self, solution):
       
        self.solution = solution
        
        # Display solution
        solution_text = "SOLUTION RESULTS\n"
        solution_text += "=" * 50 + "\n\n"
        
        solution_text += f"Method: {self.current_method}\n"
        solution_text += f"Matrix Size: {len(solution)}×{len(solution)}\n"
        
        if self.current_method in ["Jacobi", "Gauss-Seidel"]:
            iterations = len([step for step in self.iteration_data if 'error' in step])
            solution_text += f"Iterations: {iterations}\n"
            if self.iteration_data:
                final_error = self.iteration_data[-1].get('error', 'N/A')
                solution_text += f"Final Error: {final_error:.6e}\n"
        
        solution_text += "\nSolution Vector (x):\n"
        solution_text += "-" * 20 + "\n"
        
        for i, val in enumerate(solution):
            solution_text += f"x[{i+1}] = {val:12.6f}\n"
        
        # Verify solution
        residual = np.linalg.norm(self.matrix_A @ solution - self.vector_b)
        
        
        # Matrix properties
        cond_num = np.linalg.cond(self.matrix_A)
        det = np.linalg.det(self.matrix_A)
        
        solution_text += f"Matrix Determinant: {det:.2f}\n"
        
        self.solution_text.setText(solution_text)
        
        # Update final visualization
        self.matrix_animation.set_matrix(self.matrix_A, [], f"Solution completed: {self.current_method}")
        
        # Reset UI state
        self.reset_ui_state()
        self.statusBar().showMessage(f"Solution completed using {self.current_method}")
    
    def solution_error(self, error_message):
        """Handle solution errors"""
        QMessageBox.critical(self, "Solution Error", error_message)
        self.reset_ui_state()
        self.statusBar().showMessage("Solution failed")
    
    def reset_ui_state(self):
        """Reset UI to ready state"""
        self.solve_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
    
    def clear_results(self):
        """Clear all result displays"""
        self.solution_text.clear()
        self.steps_text.clear()
        self.matrix_animation.set_matrix(None)
        self.convergence_widget.set_data()
        self.iteration_data = []
    
    def save_system(self):
       
        try:
            A, b = self.get_matrix_data()
            
            data = {
                'timestamp': datetime.now().isoformat(),
                'matrix_A': A.tolist(),
                'vector_b': b.tolist(),
                'size': len(b),
                'method': self.method_combo.currentText(),
                'tolerance': self.tolerance_spinbox.value(),
                'max_iterations': self.max_iter_spinbox.value()
            }
            
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Linear System", "", "JSON Files (*.json)"
            )
            
            if filename:
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                
                QMessageBox.information(self, "Success", f"System saved to {filename}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save system: {str(e)}")
    
    def load_system(self):
        """Load system from JSON file"""
        try:
            filename, _ = QFileDialog.getOpenFileName(
                self, "Load Linear System", "", "JSON Files (*.json)"
            )
            
            if filename:
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                # Update matrix size
                size = data['size']
                self.size_spinbox.setValue(size)
                self.update_matrix_size()
                
                # Load matrix data
                A = np.array(data['matrix_A'])
                b = np.array(data['vector_b'])
                
                # Update input fields
                for i in range(size):
                    for j in range(size):
                        self.matrix_entries[i][j].setText(str(A[i, j]))
                
                for i in range(size):
                    self.vector_entries[i].setText(str(b[i]))
                
                # Update method settings
                if 'method' in data:
                    method_index = self.method_combo.findText(data['method'])
                    if method_index >= 0:
                        self.method_combo.setCurrentIndex(method_index)
                
                if 'tolerance' in data:
                    self.tolerance_spinbox.setValue(data['tolerance'])
                
                if 'max_iterations' in data:
                    self.max_iter_spinbox.setValue(data['max_iterations'])
                
                QMessageBox.information(self, "Success", f"System loaded from {filename}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load system: {str(e)}")
    
    def export_results(self):
        """Export solution results to text file"""
        if self.solution is None:
            QMessageBox.warning(self, "Warning", "No solution available to export")
            return
        
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Results", "", "Text Files (*.txt)"
            )
            
            if filename:
                with open(filename, 'w') as f:
                    f.write("LINEAR SYSTEM SOLVER RESULTS\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Method: {self.current_method}\n")
                    f.write(f"Matrix Size: {len(self.solution)}×{len(self.solution)}\n\n")
                    
                    # Write system
                    f.write("ORIGINAL SYSTEM:\n")
                    f.write("-" * 20 + "\n")
                    A, b = self.matrix_A, self.vector_b
                    for i in range(len(b)):
                        row_str = " ".join(f"{A[i,j]:8.3f}" for j in range(len(b)))
                        f.write(f"{row_str} = {b[i]:8.3f}\n")
                    
                    f.write("\nSOLUTION:\n")
                    f.write("-" * 20 + "\n")
                    for i, val in enumerate(self.solution):
                        f.write(f"x[{i+1}] = {val:12.6f}\n")
                    
                    # Write verification
                    residual = np.linalg.norm(self.matrix_A @ self.solution - self.vector_b)
                    f.write(f"\nVerification:\n")
                    f.write(f"Residual (||Ax - b||): {residual:.6e}\n")
                    
                    # Write iteration data if available
                    if self.iteration_data:
                        f.write(f"\nSOLUTION STEPS ({len(self.iteration_data)} steps):\n")
                        f.write("-" * 30 + "\n")
                        for i, step in enumerate(self.iteration_data):
                            f.write(f"Step {i+1}: {step.get('step', '')}\n")
                            if 'operation' in step:
                                f.write(f"  Operation: {step['operation']}\n")
                            if 'error' in step:
                                f.write(f"  Error: {step['error']:.6e}\n")
                            f.write("\n")
                
                QMessageBox.information(self, "Success", f"Results exported to {filename}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export results: {str(e)}")


def main():
  
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Linear System Solver")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("PyQt6 Educational Tools")
    
    # Apply modern styling
    app.setStyle('Fusion')
    
    # Create and show main window
    try:
        solver = LinearSystemSolver()
        solver.show()
        
        # Start event loop
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":

    main()


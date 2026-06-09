import tkinter as tk
from tkinter import messagebox
import numpy as np

def get_matrices():
    try:
        # Get text from the two Text widgets and convert to NumPy arrays
        mat1_str = text_matrix1.get("1.0", tk.END).strip()
        mat2_str = text_matrix2.get("1.0", tk.END).strip()
        
        if not mat1_str or not mat2_str:
            raise ValueError("Please enter both matrices")
        
        matrix1 = np.array(eval(mat1_str))
        matrix2 = np.array(eval(mat2_str))
        
        # Check if they are valid 2D arrays and same shape
        if matrix1.ndim != 2 or matrix2.ndim != 2:
            raise ValueError("Matrices must be 2-dimensional")
        if matrix1.shape != matrix2.shape:
            raise ValueError(f"Matrix shapes do not match: {matrix1.shape} vs {matrix2.shape}")
        
        return matrix1, matrix2
    except Exception as e:
        messagebox.showerror("Error", f"Invalid matrix format!\n\n{e}\n\nExample:\n[[1, 2], [3, 4]]")
        return None, None

def add_matrices():
    m1, m2 = get_matrices()
    if m1 is None:
        return
    result = m1 + m2
    show_result(result)

def subtract_matrices():
    m1, m2 = get_matrices()
    if m1 is None:
        return
    result = m1 - m2
    show_result(result)

def multiply_matrices():
    m1, m2 = get_matrices()
    if m1 is None:
        return
    try:
        result = np.dot(m1, m2)   # Matrix multiplication
        show_result(result)
    except ValueError as e:
        messagebox.showerror("Error", f"Multiplication error: {e}")

def show_result(result):
    result_window = tk.Toplevel(root)
    result_window.title("Result")
    result_text = tk.Text(result_window, height=10, width=50)
    result_text.pack(padx=10, pady=10)
    result_text.insert(tk.END, str(result))
    result_text.config(state=tk.DISABLED)

# ==================== GUI Setup ====================
root = tk.Tk()
root.title("Matrix Calculator")
root.geometry("800x600")

tk.Label(root, text="Matrix 1 (e.g. [[1,2],[3,4]])", font=("Arial", 12, "bold")).pack(pady=5)
text_matrix1 = tk.Text(root, height=8, width=60)
text_matrix1.pack(padx=10, pady=5)

tk.Label(root, text="Matrix 2 (e.g. [[5,6],[7,8]])", font=("Arial", 12, "bold")).pack(pady=5)
text_matrix2 = tk.Text(root, height=8, width=60)
text_matrix2.pack(padx=10, pady=5)

btn_frame = tk.Frame(root)
btn_frame.pack(pady=20)

tk.Button(btn_frame, text="Add", width=12, command=add_matrices).grid(row=0, column=0, padx=5)
tk.Button(btn_frame, text="Subtract", width=12, command=subtract_matrices).grid(row=0, column=1, padx=5)
tk.Button(btn_frame, text="Multiply", width=12, command=multiply_matrices).grid(row=0, column=2, padx=5)

tk.Label(root, text="Tip: Use Python list-of-lists format like [[1,2],[3,4]]", fg="gray").pack()

root.mainloop()
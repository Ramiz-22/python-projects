import customtkinter as ctk
from tkinter import messagebox
from screen_brightness_control import set_brightness, get_brightness

# --- Constants ---
# Step size for brightness adjustments when using arrow keys.
BRIGHTNESS_STEP = 5
# Small delay (in ms) before applying the brightness change after an arrow key event.
# This debounces rapid key presses into a single, final adjustment.
APPLY_CHANGES_DELAY_MS = 50 

# --- Global variables ---
# Stores the ID returned by root.after() for scheduling/cancelling the brightness application.
after_id_apply_changes = None

# --- Functions ---

def update_current_brightness_label_only():
    """Fetches the current brightness level(s) and updates the 'Current:' label in the GUI.

    Handles single and multiple display scenarios, showing an average or list if necessary.
    """
    try:
        current_levels = get_brightness()
        
        if current_levels is None:
            current_brightness_var.set("Current: N/A")
            return

        if isinstance(current_levels, list):
            if not current_levels: 
                display_text = "Current: N/A (No displays)"
            # Check if all displays have the same brightness
            elif len(set(current_levels)) == 1: 
                display_text = f"Current: {current_levels[0]}%"
            # If multiple displays have different brightness levels
            else: 
                display_text = f"Current: {', '.join(map(str, current_levels))}% (Multiple)"
        else:
            # Single display scenario (integer returned)
            display_text = f"Current: {current_levels}%"
            
        current_brightness_var.set(display_text)
        
    except Exception as e:
        # Display a generic error on the label if fetching fails
        current_brightness_var.set(f"Current: Error")


def apply_and_confirm_brightness_from_entry():
    """Reads the intended brightness value from the entry widget, applies it to the system, 
    and then updates the current brightness label to confirm the change."""
    global new_brightness_entry
    new_brightness_str = new_brightness_entry.get()
    
    try:
        new_value = int(new_brightness_str)
        
        # 1. Validate the value is within the acceptable range [0-100]
        if 0 <= new_value <= 100:
            try:
                # 2. Attempt to set the system brightness
                set_brightness(new_value)
                
                # 3. Confirm the change by updating the label
                update_current_brightness_label_only()
                
            except Exception as e:
                # Handle errors during the actual system brightness setting
                messagebox.showerror("Error Setting Brightness", f"Could not set brightness: {e}")
                
        else:
            # Handle manual input of values outside the 0-100 range
            messagebox.showerror("Invalid Input", "Brightness must be 0-100.")

    except ValueError:
        # Handle non-numeric input in the entry field
        update_current_brightness_label_only()
        messagebox.showerror("Invalid Input", "Please enter a valid number.")
        
    except Exception as e:
        # Catch any unexpected errors
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")


def schedule_apply_changes():
    """Schedules the application of the entry value after a short debounce delay.
    
    If an existing scheduled call is pending, it is cancelled and rescheduled.
    This prevents continuous system calls during rapid arrow key presses.
    """
    global after_id_apply_changes
    # Cancel any previous scheduled call
    if after_id_apply_changes:
        root.after_cancel(after_id_apply_changes)
        
    # Schedule the actual function call
    after_id_apply_changes = root.after(APPLY_CHANGES_DELAY_MS, apply_and_confirm_brightness_from_entry)


def handle_arrow_key_update_entry_only(direction):
    """Adjusts the numerical value inside the entry field based on arrow key input 
    and schedules the system brightness update."""
    global new_brightness_entry
    
    try:
        current_value_str = new_brightness_entry.get()
        
        # 1. Determine the starting value for calculation
        if not current_value_str:
            # If the entry is empty, try to derive the starting value from the 'Current' label
            current_brightness_text = current_brightness_var.get()
            try:
                # Safely parse the value from the 'Current:' label (e.g., "Current: 80%" -> 80)
                if "Current: " in current_brightness_text and "%" in current_brightness_text:
                    val_part = current_brightness_text.split("Current: ")[1].split("%")[0]
                    # Handle multiple display output (e.g., "75, 80% (Multiple)")
                    if "," in val_part: val_part = val_part.split(",")[0]
                    current_value = int(val_part)
                else: 
                    current_value = 50 # Default if parsing fails
            except: 
                current_value = 50 # Default if any other error during parsing
        else:
            # Use the value currently displayed in the entry field
            current_value = int(current_value_str)

        # 2. Calculate the new stepped value and clamp it to [0, 100]
        if direction == "up": 
            new_value = min(100, current_value + BRIGHTNESS_STEP)
        elif direction == "down": 
            new_value = max(0, current_value - BRIGHTNESS_STEP)
        else: 
            return # Ignore other keys

        # 3. Update the entry field display
        new_brightness_entry.delete(0, ctk.END)
        new_brightness_entry.insert(0, str(new_value))

        # 4. Schedule the actual system brightness update
        schedule_apply_changes()

    except ValueError:
        # If the entry contains non-numeric text, reset it to a default and schedule update
        new_brightness_entry.delete(0, ctk.END)
        new_brightness_entry.insert(0, "50")
        schedule_apply_changes()
        
    except Exception as e:
        # General error handling during key press logic
        print(f"Error with arrow key handling: {e}") # Debugging output


def initial_load_brightness():
    """Initializes the entry field and the 'Current:' label on application startup."""
    global new_brightness_entry
    value_to_set_in_entry = 50 # Default starting value
    
    try:
        current_levels = get_brightness()
        
        if current_levels is not None:
            # Set entry value based on the first display's current brightness
            if isinstance(current_levels, list):
                if current_levels: value_to_set_in_entry = current_levels[0]
            else: 
                value_to_set_in_entry = current_levels

        # Update the entry widget
        if new_brightness_entry:
            new_brightness_entry.delete(0, ctk.END)
            new_brightness_entry.insert(0, str(value_to_set_in_entry))
            
        # Update the 'Current:' label
        update_current_brightness_label_only()
        
    except Exception:
        # Fallback if fetching brightness fails completely
        if new_brightness_entry:
            new_brightness_entry.delete(0, ctk.END)
            new_brightness_entry.insert(0, str(value_to_set_in_entry))
        current_brightness_var.set("Current: Error on load")


# --- GUI Setup ---
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("Screen Brightness Control")
# Reduced height due to removal of status bar
root.geometry("380x190") 
root.resizable(False, False)

# Main container frame
main_frame = ctk.CTkFrame(root, corner_radius=15)
main_frame.pack(expand=True, fill="both", padx=10, pady=10)

# 1. Current Brightness Label
current_brightness_var = ctk.StringVar()
current_brightness_label = ctk.CTkLabel(main_frame, textvariable=current_brightness_var, font=("Arial", 14))
current_brightness_label.pack(pady=(20, 10))

# 2. Input Frame (Label + Entry)
input_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
input_frame.pack(pady=10)

ctk.CTkLabel(input_frame, text="Set Brightness (0-100):", font=("Arial", 12)).pack(side="left", padx=(0, 10))
new_brightness_entry = ctk.CTkEntry(input_frame, width=60, font=("Arial", 12), corner_radius=8)
new_brightness_entry.pack(side="left")

# Bindings for entry widget
# Enter key triggers immediate brightness set
new_brightness_entry.bind("<Return>", lambda event: apply_and_confirm_brightness_from_entry())
# Arrow keys adjust value in entry and schedule the set (with debounce)
new_brightness_entry.bind("<KeyPress-Up>", lambda event: handle_arrow_key_update_entry_only("up"))
new_brightness_entry.bind("<KeyPress-Down>", lambda event: handle_arrow_key_update_entry_only("down"))

# 3. Set Button
set_button = ctk.CTkButton(main_frame, text="Set Brightness", command=apply_and_confirm_brightness_from_entry, font=("Arial", 12, "bold"), corner_radius=8)
set_button.pack(pady=(15, 20))

# --- Application Start ---
initial_load_brightness()
new_brightness_entry.focus()

root.mainloop()
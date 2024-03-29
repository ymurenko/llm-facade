import os
import threading
import time
import dearpygui.dearpygui as dpg
from util import get_gpu_usage, get_cpu_usage, get_system_info
from conversation import Conversation, wrap_text
from model_manager import ModelManager
from style.styles import TextColor

num_messages = 0

def render_chat_messages():
    dpg.set_value("raw_output_area", conversation.get_raw_string_for_display())
    dpg.delete_item(output_area, children_only=True)

    for message in conversation.conversation_history:
        content = str(message["content"])
        formatted_content = wrap_text(content, conversation.line_width)
        role = message["role"]
        height = 30 + (formatted_content.count('\n') * 13)
        if role == "user":
            with dpg.group(horizontal=True, parent=output_area):
                dpg.add_text("User:")
                user_message = dpg.add_input_text(default_value=formatted_content, multiline=True, readonly=True, height=height, width=-2)
                dpg.bind_item_theme(user_message, user_message_theme)
        elif role == "ai":
            with dpg.group(horizontal=True, parent=output_area):
                dpg.add_text("  AI:")
                ai_mesasge = dpg.add_input_text(default_value=formatted_content, multiline=True, readonly=True, height=height, width=-2)
                dpg.bind_item_theme(ai_mesasge, ai_message_theme)

def render_streamed_ai_message():
    children = dpg.get_item_children(output_area, slot=1)
    dpg.set_value("raw_output_area", conversation.get_raw_string_for_display())
    if len(children) == len(conversation.conversation_history) - 1:
        text_area = dpg.get_item_children(children[-1], slot=1)[-1]
        content = str(conversation.get_last_ai_message())
        formatted_content = wrap_text(content, conversation.line_width)
        height = 30 + (formatted_content.count('\n') * 13)
        dpg.configure_item(text_area, default_value=formatted_content, height=height)
    else:
        render_chat_messages()

def viewport_y(offset: int = 0):
    """
    Returns height relative to viewport height
    """
    return dpg.get_viewport_height() + offset

def viewport_x(offset: int = 0):
    """
    Used for setting width relative to viewport
    """
    return dpg.get_viewport_width() + offset

def set_token_output(value: str):
    """
    Adds incoming tokens to conversation and displays in chat window
    """
    conversation.append_partial_ai_message(value)
    render_streamed_ai_message()

def set_inference_speed_info(value: float):
    """
    Displays the inference speed in tokens/s
    """
    dpg.set_value("inference_speed_info", f"Speed: {value:.1f} tokens/s")

def set_context_length_info(value: int, max: int):
    """
    Displays the current context length
    """
    dpg.set_value("context_length_info", f"Context: {value}/{max} tokens")

def on_done_loading_model():
    """
    Callback for when the model process is started and the model is loaded
    """
    dpg.configure_item("inference_btn", enabled=True)
    set_system_message(f"Loaded '{selected_model}'. Ready.", TextColor.GREEN)

def on_done_unloading_model():
    """
    Callback for when the model process is stopped and the model is unloaded
    """
    dpg.configure_item("load_model_btn", enabled=True)
    set_system_message(f"No model loaded.", TextColor.RED)
    
def on_load_model():
    """
    Start model process and load model
    """
    dpg.configure_item("load_model_btn", enabled=False)
    dpg.configure_item("unload_model_btn", enabled=True)
    set_system_message(f"Loading model '{selected_model}'...", TextColor.YELLOW)
    llm_process_manager.start_model_worker()

def on_unload_model():
    """
    Stop model process and unload model
    """
    dpg.configure_item("unload_model_btn", enabled=False)
    dpg.configure_item("inference_btn", enabled=False)
    set_system_message(f"Unloading model '{selected_model}'...", TextColor.YELLOW)
    llm_process_manager.stop_model_worker()

def on_send_inference():
    """
    Add user message to conversation and add raw conversation string
    to the input queue. Then run inference.
    """
    if dpg.get_item_configuration("inference_btn")["enabled"] == False:
        return
    
    input = dpg.get_value("chat_input_area")
    if input is not None and input != "":
        conversation.append_full_user_message(input)
    render_chat_messages()
    input_string = conversation.get_string_for_inference()
    dpg.set_value("chat_input_area", "")
    dpg.configure_item("inference_btn", enabled=False)
    set_system_message(f"Inferencing...", TextColor.BLUE)
    llm_process_manager.run_inference(input_string)
    dpg.configure_item("inference_btn", enabled=True)
    set_system_message(f"Inference complete. Ready.", TextColor.GREEN)

def on_select_model(sender):
    """
    Set selected model and update max tokens slider with
    the new max content length
    """
    global selected_model
    selected_model = dpg.get_value(sender)
    llm_process_manager.set_model(selected_model)
    dpg.configure_item("max_tokens_slider", max_value=get_max_token_slider_limit())

def on_set_device_map(sender):
    """
    Set device map (converted to suitable string for hf)
    """
    selected_device = dpg.get_value(sender)
    if selected_device == "Auto":
        llm_process_manager.device_map = 'auto'
    elif selected_device == "GPU":
        llm_process_manager.device_map = "cuda"
    elif selected_device == "CPU":
        llm_process_manager.device_map = "cpu"
    elif selected_device == "MPS":
        llm_process_manager.device_map = "mps"

def on_set_temperature(sender):
    llm_process_manager.temperature = dpg.get_value(sender)

def on_set_top_p(sender):
    llm_process_manager.top_p = dpg.get_value(sender)

def on_set_top_k(sender):
    llm_process_manager.top_k = dpg.get_value(sender)

def on_set_max_tokens(sender):
    llm_process_manager.max_new_tokens = dpg.get_value(sender)

def get_max_token_slider_limit():
    """
    Determines the max value for the max tokens slider
    """
    max_context_length = llm_process_manager.max_context_length
    if max_context_length is int:
        return max_context_length
    return 4096

def on_set_system_prompt():
    """
    Set the system prompt and update the chat output display
    """
    system_prompt = dpg.get_value("system_prompt_input")
    conversation.set_system_prompt(system_prompt)
    render_chat_messages()

def set_system_message(message: str, color: TextColor):
    """
    Sets system info message and color
    """
    dpg.configure_item("system_message_display", default_value=message, color=color.value)

def on_use_multiline_input(sender):
    """
    Toggle whether to use multiline input area for chat
    """
    if dpg.get_value(sender):
        dpg.configure_item("chat_input_area", multiline=True)
        dpg.configure_item("chat_input_window", height=100)
        dpg.configure_item("chat_output_window", height=-108)
    else:
        dpg.configure_item("chat_input_area", multiline=False)
        dpg.configure_item("chat_input_window", height=40)
        dpg.configure_item("chat_output_window", height=-48)

def on_reset_conversation():
    """
    Reset conversation history and display
    """
    conversation.reset_conversation()
    render_chat_messages()

def on_remove_last_message():
    """
    Remove last message from conversation history and display
    """
    conversation.remove_last_message()
    render_chat_messages()

def on_set_line_width(sender):
    """
    Set line width for chat output
    """
    conversation.line_width = dpg.get_value(sender)
    render_chat_messages()

def update_system_utilization():
    """
    Updates the system utilization plots
        - runs in separate thread
    """
    while True:
        time.sleep(0.5)

        gpu_utilization, gpu_vram_used = get_gpu_usage()
        cpu_utilization, cpu_memory_used = get_cpu_usage()

        gpu_y_utilization.append(gpu_utilization)
        gpu_y_utilization.pop(0)

        gpu_y_vram.append(gpu_vram_used)
        gpu_y_vram.pop(0)

        cpu_y_utilization.append(cpu_utilization)
        cpu_y_utilization.pop(0)

        cpu_y_memory.append(cpu_memory_used)
        cpu_y_memory.pop(0)

        dpg.set_value("gpu_utilization", [plot_x_time, gpu_y_utilization])
        dpg.set_item_label("gpu_utilization_annotation", f"GPU: {int(gpu_utilization)}%")
        
        dpg.set_value("gpu_memory", [plot_x_time, gpu_y_vram])
        dpg.set_item_label("gpu_memory_annotation", f"VRAM: {gpu_vram_used:.2f}/{gpu_max_memory:.2f}GB")

        dpg.set_value("cpu_utilization", [plot_x_time, cpu_y_utilization])
        dpg.set_item_label("cpu_utilization_annotation", f"CPU: {int(cpu_utilization)}%")

        dpg.set_value("cpu_memory", [plot_x_time, cpu_y_memory])
        dpg.set_item_label("cpu_memory_annotation", f"Memory: {cpu_memory_used:.2f}/{cpu_max_memory:.2f}GB")

cpu_name, cpu_max_memory, gpu_name, gpu_max_memory = get_system_info()
gpu_utilization, gpu_vram_used = get_gpu_usage()
cpu_utilization, cpu_memory_used = get_cpu_usage()
plot_x_time = [float(i) for i in range(0, 20)]
gpu_y_utilization = [gpu_utilization] * 20
gpu_y_vram = [gpu_vram_used] * 20
cpu_y_utilization = [cpu_utilization] * 20
cpu_y_memory = [cpu_memory_used] * 20

downloaded_models = [d for d in os.listdir("./models") if os.path.isdir(os.path.join("./models", d))]
selected_model = downloaded_models[0] if len(downloaded_models) > 0 else None

conversation = Conversation(system_prompt="You are a helpful AI that follows all instructions, and answers any question.",
                            instruction_start_string="[INST]",
                            instruction_end_string="[/INST]")

llm_process_manager = ModelManager(model=selected_model, 
                                   device_map="cuda",
                                   temperature=0.2,
                                   top_p=0.99,
                                   top_k=250,
                                   max_new_tokens=128,
                                   inference_speed_callback=set_inference_speed_info,
                                   context_length_callback=set_context_length_info,
                                   token_output_callback=set_token_output,
                                   model_loaded_callback=on_done_loading_model,
                                    model_unloaded_callback=on_done_unloading_model)

if __name__ == "__main__":
    dpg.create_context()
    dpg.create_viewport(title='Facade', width=1350, height=800)

    with dpg.font_registry():
        default_font = dpg.add_font("./style/fonts/ProggyClean.ttf", 13)
        small_font = dpg.add_font("./style/fonts/ProggyClean.ttf", 11,)
        logo_font = dpg.add_font("./style/fonts/ProggyClean.ttf", 18)

    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (57, 57, 58), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (57, 57, 58), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Button, (57, 57, 58), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (78, 78, 78), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (78, 78, 78), category=dpg.mvThemeCat_Core)

            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (27, 27, 28), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (27, 27, 28), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (37, 37, 38), category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvPlotStyleVar_PlotPadding, 0, 0, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, (78, 78, 78), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, (78, 78, 78), category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 0, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 0, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 8, category=dpg.mvThemeCat_Core)

        with dpg.theme_component(dpg.mvButton, enabled_state=False):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (131, 131, 131), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Button, (47, 47, 48), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (47, 47, 48), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (47, 47, 48), category=dpg.mvThemeCat_Core)
    
    with dpg.theme() as label_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 1, category=dpg.mvThemeCat_Core)

    with dpg.theme() as settings_tab_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (57, 57, 58), category=dpg.mvThemeCat_Core)

    # with dpg.theme() as slider_theme:
    #     with dpg.theme_component(dpg.mvAll):
    #         dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 4, 1, category=dpg.mvThemeCat_Core)

    with dpg.theme() as user_message_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (27, 27, 28), category=dpg.mvThemeCat_Core)

    with dpg.theme() as ai_message_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (37, 37, 38), category=dpg.mvThemeCat_Core)

    with dpg.theme() as data_group_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 4, 4, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (27, 27, 28), category=dpg.mvThemeCat_Core)
            
    with dpg.theme() as inference_stats_group_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 8, 5, category=dpg.mvThemeCat_Core)

    with dpg.theme() as output_area_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10, 8, category=dpg.mvThemeCat_Core)

    with dpg.theme() as text_input_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (57, 57, 58), category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 5, category=dpg.mvThemeCat_Core)

    with dpg.theme() as second_column_theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (57, 57, 58), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (151, 151, 151), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (20 , 119, 200, 230), category=dpg.mvThemeCat_Core)

    dpg.bind_theme(global_theme)
    dpg.bind_font(default_font)

    with dpg.window(tag="primary_window", width=1350, height=-1) as window:
        with dpg.group(horizontal=True) as primary_group:
            # FIRST COLUMN
            with dpg.child_window(width=-346, height=-1, border=False):
                with dpg.child_window(tag="plots", height=60):
                    with dpg.group(horizontal=True) as data_group:
                        dpg.bind_item_theme(data_group, data_group_theme)
                        # LOGO
                        with dpg.child_window(width=-806, height=44) as logo_window:
                            dpg.add_text(f" Facade v0.1 ", pos=(6, 10))
                            dpg.bind_item_font(logo_window, logo_font)

                        # GPU UTILIZATION PLOT
                        with dpg.plot(height=44, width=197) as gpu_utilization_plot:
                            dpg.add_plot_axis(dpg.mvXAxis, tag="gpu_utilization_x_axis", no_tick_labels=True)
                            dpg.add_plot_axis(dpg.mvYAxis, tag="gpu_utilization_y_axis")
                            dpg.set_axis_ticks("gpu_utilization_x_axis", (("", 0), ("", 100)))
                            dpg.set_axis_ticks("gpu_utilization_y_axis", (("", 0), ("", 100)))
                            dpg.set_axis_limits("gpu_utilization_x_axis", 0, 19)
                            dpg.set_axis_limits("gpu_utilization_y_axis", -2, 100)

                            dpg.add_plot_annotation(label=f"GPU: {int(gpu_utilization)}%", offset=(2, -100), tag="gpu_utilization_annotation")
                            dpg.add_line_series(plot_x_time, gpu_y_utilization, parent="gpu_utilization_y_axis", tag="gpu_utilization")

                        # GPU MEMORY PLOT
                        with dpg.plot(height=44, width=197) as gpu_memory_plot:
                            dpg.add_plot_axis(dpg.mvXAxis, tag="gpu_memory_x_axis", no_tick_labels=True)
                            dpg.add_plot_axis(dpg.mvYAxis, tag="gpu_memory_y_axis")
                            dpg.set_axis_ticks("gpu_memory_x_axis", (("", 0), ("", 100)))
                            dpg.set_axis_ticks("gpu_memory_y_axis", (("", 0), ("", 100)))
                            dpg.set_axis_limits("gpu_memory_x_axis", 0, 19)
                            dpg.set_axis_limits("gpu_memory_y_axis", 0, (gpu_max_memory))

                            dpg.add_plot_annotation(label=f"VRAM: {gpu_vram_used:.2f}/{gpu_max_memory:.2f}GB", offset=(2, -100), tag="gpu_memory_annotation")
                            dpg.add_line_series(plot_x_time, gpu_y_vram, parent="gpu_memory_y_axis", tag="gpu_memory")

                        # CPU UTILIZATION PLOT
                        with dpg.plot(height=44, width=197) as cpu_utilization_plot:
                            dpg.add_plot_axis(dpg.mvXAxis, tag="cpu_utilization_x_axis", no_tick_labels=True)
                            dpg.add_plot_axis(dpg.mvYAxis, tag="cpu_utilization_y_axis")
                            dpg.set_axis_ticks("cpu_utilization_x_axis", (("", 0), ("", 100)))
                            dpg.set_axis_ticks("cpu_utilization_y_axis", (("", 0), ("", 100)))
                            dpg.set_axis_limits("cpu_utilization_x_axis", 0, 19)
                            dpg.set_axis_limits("cpu_utilization_y_axis", -2, 100)

                            dpg.add_plot_annotation(label=f"CPU: {int(cpu_utilization)}%", offset=(2, -100), tag="cpu_utilization_annotation")
                            dpg.add_line_series(plot_x_time, cpu_y_utilization, parent="cpu_utilization_y_axis", tag="cpu_utilization")
                        
                        # CPU MEMORY PLOT
                        with dpg.plot(height=44, width=197) as cpu_memory_plot:
                            dpg.add_plot_axis(dpg.mvXAxis, tag="cpu_memory_x_axis", no_tick_labels=True)
                            dpg.add_plot_axis(dpg.mvYAxis, tag="cpu_memory_y_axis")
                            dpg.set_axis_ticks("cpu_memory_x_axis", (("", 0), ("", 100)))
                            dpg.set_axis_ticks("cpu_memory_y_axis", (("", 0), ("", 100)))
                            dpg.set_axis_limits("cpu_memory_x_axis", 0, 19)
                            dpg.set_axis_limits("cpu_memory_y_axis", 0, (cpu_max_memory))

                            dpg.add_plot_annotation(label=f"Memory: {cpu_memory_used:.2f}/{cpu_max_memory:.2f}GB", offset=(2, -100), tag="cpu_memory_annotation")
                            dpg.add_line_series(plot_x_time, cpu_y_memory, parent="cpu_memory_y_axis", tag="cpu_memory")

                # INFO BLOCK GROUP
                with dpg.group(horizontal=True) as inference_stats_group:
                    # SYSTEM MESSAGE DISPLAY
                    with dpg.child_window(width=-488, height=30):
                        with dpg.group(horizontal=True):
                            dpg.add_text(tag="system_message_display", default_value="No model loaded.", color=TextColor.RED.value)

                    # TOKEN SPEED INFO
                    with dpg.child_window(width=236, height=30):
                        with dpg.group(horizontal=True):
                            dpg.add_text(tag="inference_speed_info", default_value="Speed: - tokens/s")

                    # CONTEXT LENGTH INFO
                    with dpg.child_window(width=236, height=30):
                        with dpg.group(horizontal=True):
                            dpg.add_text(tag="context_length_info", default_value="Context: -/- tokens")

                    dpg.bind_item_theme(inference_stats_group, inference_stats_group_theme)

                # TABS
                with dpg.tab_bar() as tabs:
                    # CHAT TAB
                    with dpg.tab(label="Chat"):
                        # OUTPUT AREA
                        with dpg.child_window(tag="chat_output_window", autosize_x=False, height=-48) as output_area: 
                            dpg.bind_item_theme(output_area, output_area_theme)

                        # INPUT AREA
                        with dpg.child_window(tag="chat_input_window", height=40):
                            with dpg.group(horizontal=True) as input_area:
                                dpg.bind_item_theme(input_area, text_input_theme)
                                dpg.add_input_text(tag="chat_input_area", multiline=False, on_enter=True, height=-2, width=-90, callback=on_send_inference)
                                dpg.add_button(label="Inference", tag="inference_btn", callback=on_send_inference, width=80, height=-1, enabled=False)
                    # RAW TAB
                    with dpg.tab(label="Raw"):
                        with dpg.child_window(autosize_x=False, height=-48): 
                            dpg.add_input_text(tag="raw_output_area",  multiline=True, readonly=False, tracked=True, track_offset=1.0, height=-2, width=-2)

                        # INPUT AREA
                        with dpg.child_window(height=40):
                            with dpg.group(horizontal=True) as raw_input_area:
                                dpg.bind_item_theme(raw_input_area, text_input_theme)
                                dpg.add_spacer(height=-2, width=-90)
                                dpg.add_button(label="Inference", width=80, height=-1, enabled=False)

                    # SETTINGS TAB
                    with dpg.tab(label="Prompt settings") as settings_tab:
                        dpg.bind_item_theme(settings_tab, settings_tab_theme)
                        with dpg.child_window(height=-4):
                            dpg.add_text(default_value="System prompt")
                            with dpg.group(horizontal=True):
                                    dpg.add_input_text(tag="system_prompt_input", multiline=True, height=80, width=-90, default_value=conversation.get_system_prompt())
                                    dpg.add_button(label="Set", width=80, height=80, callback=on_set_system_prompt)
                                    # TODO: input wrapper settings
                                    # TODO: newline settings

            # SECOND COLUMN
            with dpg.group() as second_column:
                dpg.bind_item_theme(second_column, second_column_theme)

                with dpg.child_window(tag="hyperparameters_window", width=340, height=-119) as hyperparameters_window:
                    # MODEL LOADING
                    dpg.add_combo(default_value=selected_model, items=downloaded_models, width=-10, callback=on_select_model)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Device map:")
                        dpg.add_radio_button(label="device_map", items=['Auto', 'GPU', 'CPU', 'MPS'], horizontal=True, default_value='GPU', callback=on_set_device_map)
                        # TODO: Add load_in_8bit option
                        # TODO: Add low_cpu_memory_usage option

                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Load model", tag="load_model_btn", width=155, callback=on_load_model, enabled=True)
                        dpg.add_button(label="Unload model", tag="unload_model_btn", width=155, callback=on_unload_model, enabled=False)

                    # HYPERPARAMETERS
                    dpg.add_separator()
                    with dpg.group() as hyperparameters_group:
                        temp_label = dpg.add_text("Temperature")
                        dpg.bind_item_theme(temp_label, label_theme)
                        dpg.add_slider_float(callback=on_set_temperature, tracked=True, default_value=0.2, min_value=0.0, max_value=2.0, width=-4)
                        top_p_label = dpg.add_text("Top P")
                        dpg.bind_item_theme(top_p_label, label_theme)
                        dpg.add_slider_float(callback=on_set_top_p, tracked=True, default_value=0.99, min_value=0.00, max_value=1.00, width=-4)
                        top_k_label = dpg.add_text("Top K")
                        dpg.bind_item_theme(top_k_label, label_theme)
                        dpg.add_slider_int(callback=on_set_top_k, tracked=True, default_value=250, min_value=0, max_value=1000, width=-4)
                        max_tokens_label = dpg.add_text("Max tokens to generate")
                        dpg.bind_item_theme(max_tokens_label, label_theme)
                        dpg.add_slider_int(tag="max_tokens_slider", callback=on_set_max_tokens, tracked=True, default_value=128, min_value=0, max_value=get_max_token_slider_limit(), width=-4)
                    dpg.add_separator()

                # CHAT SETTINGS
                with dpg.child_window(height=110):
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Remove last message", width=155, callback=on_remove_last_message)
                        dpg.add_button(label="Reset conversation", width=155, callback=on_reset_conversation)

                    dpg.add_checkbox(label="Use multiline input", default_value=False, callback=on_use_multiline_input)
                    line_width_label = dpg.add_text("Output line width")
                    dpg.bind_item_theme(line_width_label, label_theme)
                    dpg.add_slider_int(callback=on_set_line_width, tracked=True, default_value=conversation.line_width, min_value=50, max_value=300, width=-4)
                        # TODO: Manual edit mode
                
        
    gpu_stats_thread = threading.Thread(target=update_system_utilization)
    gpu_stats_thread.daemon = True
    gpu_stats_thread.start()

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.show_style_editor()
    dpg.set_primary_window("primary_window", True)
    dpg.start_dearpygui()

    llm_process_manager.stop_model_worker()

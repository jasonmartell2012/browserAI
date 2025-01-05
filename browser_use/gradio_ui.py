import gradio as gr

def create_gradio_interface():
    def on_button_click(input_text):
        return f"Button clicked with input: {input_text}"

    with gr.Blocks() as demo:
        text_input = gr.Textbox(label="Enter text")
        button = gr.Button("Submit")
        output = gr.Textbox(label="Output")

        button.click(fn=on_button_click, inputs=text_input, outputs=output)

    demo.launch()

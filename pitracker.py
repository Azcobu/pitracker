from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
import matplotlib.pyplot as plt
from kivy_garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg

# Placeholder for Matplotlib graph
def generate_placeholder_graph():
    plt.figure(figsize=(5, 3))
    plt.plot([0, 1, 2, 3], [0, 1, 4, 9], label='Placeholder')
    plt.title('Daily Temperature Graph')
    plt.legend()
    return plt.gcf()

class MainPage(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', spacing=10, padding=20)
        
        layout.add_widget(Label(text="Select Graph Type", font_size=24))
        
        daily_button = Button(text="Daily Temperatures", font_size=20, size_hint=(1, 0.2))
        daily_button.bind(on_press=self.open_daily_temp)
        layout.add_widget(daily_button)
        
        for title in ["Weekly Temperatures", "Yearly Temperatures", "Stock Market Info"]:
            btn = Button(text=title, font_size=20, size_hint=(1, 0.2))
            btn.bind(on_press=self.placeholder_page)
            layout.add_widget(btn)
        
        self.add_widget(layout)
    
    def open_daily_temp(self, instance):
        self.manager.current = "daily_temp"
    
    def placeholder_page(self, instance):
        self.manager.current = "placeholder"

class DailyTempPage(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical', spacing=10, padding=20)
        
        self.graph_widget = FigureCanvasKivyAgg(generate_placeholder_graph())
        self.layout.add_widget(self.graph_widget)
        
        back_button = Button(text="Back to Main", font_size=20, size_hint=(1, 0.2))
        back_button.bind(on_press=self.go_back)
        self.layout.add_widget(back_button)
        
        self.add_widget(self.layout)
        Clock.schedule_interval(self.update_graph, 60)  # Update every minute
    
    def go_back(self, instance):
        self.manager.current = "main"
    
    def update_graph(self, dt):
        self.layout.remove_widget(self.graph_widget)
        self.graph_widget = FigureCanvasKivyAgg(generate_placeholder_graph())
        self.layout.add_widget(self.graph_widget, index=0)

class PlaceholderPage(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', spacing=10, padding=20)
        
        layout.add_widget(Label(text="Placeholder Page", font_size=24))
        back_button = Button(text="Back to Main", font_size=20, size_hint=(1, 0.2))
        back_button.bind(on_press=self.go_back)
        layout.add_widget(back_button)
        
        self.add_widget(layout)
    
    def go_back(self, instance):
        self.manager.current = "main"

class MyApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(MainPage(name="main"))
        sm.add_widget(DailyTempPage(name="daily_temp"))
        sm.add_widget(PlaceholderPage(name="placeholder"))
        return sm

if __name__ == "__main__":
    MyApp().run()

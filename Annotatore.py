from kivy.app import App
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.graphics import Color, Line
from kivy.core.window import Window
from plyer import filechooser
from os.path import splitext


class ImmagineToccabile(Image):
    def salva(self, instance=None):
        file = open(f"{splitext(self.source)[0]}.txt", "x")
        for asc, ordin in self.posizioni:
            img_w, img_h = self.get_norm_image_size()
            file.write(f"0 {asc / img_w:.6f} {ordin / img_h:.6f} {20 / img_w:.6f} {20 / img_h:.6f}\n")
            print(f"0 {asc / img_w:.6f} {ordin / img_h:.6f} {20 / img_w:.6f} {20 / img_h:.6f}\n")
        file.close()
        self.costruzione()

    def __init__(self, pulsante, costruzione, applicazione, **kwargs):
        super().__init__(**kwargs)
        self.applicazione = applicazione
        self.posizioni = set()
        self.costruzione = costruzione
        pulsante.bind(on_press=self.salva)

    def on_touch_down(self, touch, *args):
        if self.collide_point(*touch.pos):
            # Convert to local image-relative coordinates
            local_x = touch.x
            local_y = touch.y

            self.posizioni.add((local_x, local_y))

            with self.canvas:
                from kivy.graphics import Color, Line
                Color(0, 1, 0, 1)
                half = 10  # half of 40px
                Line(rectangle=(local_x - half, local_y - half, 20, 20), width=1.5)

            return True
        return False


class AnnotatoreApp(App):
    def ritorna(self, instance=None):
        self.cornice.remove_widget(self.confermatore)
        self.cornice.remove_widget(self.immagine)
        self.cornice.add_widget(self.sceglitore)

    def seleziona(self, selezionato):
        if selezionato:
            self.fonte = selezionato[0]
            self.cornice.remove_widget(self.sceglitore)
            self.confermatore = Button(text="Conferma")
            self.confermatore.pos_hint = {"top": 1, "right": 1}
            self.confermatore.size_hint = (1, .1)
            self.cornice.add_widget(self.confermatore)
            self.immagine = ImmagineToccabile(self.confermatore, self.ritorna, self, source=self.fonte)
            self.immagine.pos_hint = {"top": .9, "right": 1}
            self.cornice.add_widget(self.immagine)

    def aprisceglitore(self, instance):
        filechooser.open_file(on_selection=self.seleziona)

    def build(self, instance=None):
        self.cornice = RelativeLayout()
        self.sceglitore = Button(text="Scegli immagine")
        self.sceglitore.bind(on_press=self.aprisceglitore)
        self.cornice.add_widget(self.sceglitore)
        return self.cornice


if __name__ == "__main__":
    AnnotatoreApp().run()

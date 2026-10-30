import pygame

import vars as V
import control as C

import classes as classes

#region show fps
pygame.font.init()

font = pygame.font.SysFont('Arial', 15)

def show_fps(screen, font, clock: pygame.time.Clock, color=V.BLACK):
    fps = int(clock.get_fps())
    fps_text = f"FPS: {fps}"
    fps_surface = font.render(fps_text, True, color)
    rect = fps_surface.get_rect()
    rect.topright = (V.WIDTH, 0)
    screen.blit(fps_surface, rect.topleft)
#endregion


def draw_background(screen, color):
    screen.fill(color)

def draw_cart(screen, cart: classes.Cart):
    pygame.draw.rect(screen, cart.color, pygame.Rect(cart.x-cart.width // 2, cart.y-cart.height // 2, cart.width, cart.height))

def draw_pole(screen, pole: classes.Pole):
    """
    p1-p4: clockwise from connector (i.e. p4 is first anti-clockwise from connector)
    """

    p1 = C.rotate((pole.x - pole.width // 2, pole.y), (pole.x, pole.y), pole.theta)
    p2 = C.rotate((pole.x - pole.width // 2, pole.y + pole.length), (pole.x, pole.y), pole.theta)
    p3 = C.rotate((pole.x + pole.width // 2, pole.y + pole.length), (pole.x, pole.y), pole.theta)
    p4 = C.rotate((pole.x + pole.width // 2, pole.y), (pole.x, pole.y), pole.theta) 
    
    pygame.draw.polygon(screen, pole.color, (p1, p2, p3, p4))
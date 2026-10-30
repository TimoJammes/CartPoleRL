import numpy as np
import pygame

import classes 
import constants as Co
import vars as V

def rotate(p: tuple, center_p: tuple, theta: float) -> np.ndarray:
    p = np.array(p)
    center_p = np.array(center_p)

    cos = np.cos(theta)
    sin = np.sin(theta)

    rot = np.array([[cos, sin],
                    [-sin, cos]])
    return ((p-center_p) @ rot) + center_p


def update_pole(pole: classes.Pole, cart: classes.Cart):
    omega_dot = - Co.g / pole.length * np.sin(pole.theta) + cart.curr_acc / pole.length * np.cos(pole.theta) - Co.pole_friction_coeff * pole.omega
    pole.omega += omega_dot

    pole.theta += pole.omega

    pole.x = cart.x

    # pole.theta = pole.theta % np.pi
    pole.theta = (pole.theta + np.pi) % (2 * np.pi) - np.pi

    # if abs(pole.theta) < 1e-3 and abs(pole.omega) < 

def update_cart(cart: classes.Cart, playerInput: int):

    cart.curr_acc = playerInput * cart.acc_norm - Co.cart_friction_coeff * cart.velocity
    
    cart.velocity += cart.curr_acc

    cart.x += cart.velocity

    if cart.x < cart.width:
        cart.x = cart.width
        # cart.acceleration = 0
        cart.velocity = 0
        cart.curr_acc = 0

    elif V.WIDTH - cart.x < cart.width:
        cart.x = V.WIDTH - cart.width
        cart.velocity = 0
        # cart.acceleration = 0
        cart.velocity = 0
        cart.curr_acc = 0


def reset(cart: classes.Cart, pole: classes.Pole):
    cart.x = V.WIDTH // 2
    cart.velocity = 0
    cart.curr_acc = 0

    pole.x = cart.x
    pole.theta = np.deg2rad(V.rng.uniform(179, 181))
    pole.theta = (pole.theta + np.pi) % (2 * np.pi) - np.pi
    pole.omega = 0
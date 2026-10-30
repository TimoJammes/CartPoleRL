import pygame
import numpy as np

import vars as V
import display as D
import classes as classes
import control as C
import agent as A

cart = classes.Cart(V.WIDTH//2, V.HEIGHT//2, 0, 3, 100, 50, V.BLACK)
pole = classes.Pole(cart.x, cart.y, np.pi*99/100, 0, 10, 200, V.RED)

min_pole_theta = np.inf
max_pole_theta =  -np.inf
min_pole_omega = np.inf
max_pole_omega = -np.inf
min_cart_x = np.inf
max_cart_x = -np.inf
min_cart_vel = np.inf
max_cart_vel = -np.inf

agent = A.Agent()

do_agent = False
limit_max_angle = False
do_display = True

episode = 1

mainloop=True
while mainloop:
    if do_display:
        V.clock.tick(V.FPS)

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            mainloop = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                do_display = not(do_display)

            if event.key == pygame.K_r:
                C.reset(cart, pole)



    if do_agent:
        agent_input = agent.get_action(pole.theta, pole.omega, cart.x, cart.velocity)
    else:
        keys = pygame.key.get_pressed()
        user_input = keys[pygame.K_RIGHT]-keys[pygame.K_LEFT]
        agent_input = user_input

    C.update_cart(cart, agent_input)
    C.update_pole(pole, cart)

    is_fallen_past_angle = 180 - np.rad2deg(abs(pole.theta)) > 20

    if do_agent:
        agent.update_Q(pole.theta, pole.omega, cart.x, cart.velocity, is_fallen_past_angle)

    if do_display:
        D.draw_background(V.screen, V.WHITE)
        D.draw_cart(V.screen, cart)
        D.draw_pole(V.screen, pole)
        pygame.display.flip()

    if limit_max_angle and is_fallen_past_angle:
        C.reset(cart, pole)

        # print(min_pole_omega, max_pole_omega)
        # min_pole_omega = np.inf
        # max_pole_omega = - np.inf
        episode += 1
        print(f"Episode: {episode}")
        print(f"Starting angle: {np.rad2deg(pole.theta)}")
    min_pole_theta = min(min_pole_theta, pole.theta)
    max_pole_theta = max(max_pole_theta, pole.theta)
    min_pole_omega = min(min_pole_omega, pole.omega)
    max_pole_omega = max(max_pole_omega, pole.omega)
    min_cart_x = min(min_cart_x, cart.x)
    max_cart_x = max(max_cart_x, cart.x)
    min_cart_vel = min(min_cart_vel, cart.velocity)
    max_cart_vel = max(max_cart_vel, cart.velocity)

print(min_pole_theta, max_pole_theta)
print(min_pole_omega, max_pole_omega)
print(min_cart_x, max_cart_x)
print(min_cart_vel, max_cart_vel)

pygame.quit()

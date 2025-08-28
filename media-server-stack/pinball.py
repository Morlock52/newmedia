import pygame
import math
import random

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Space Alien AI Heist Pinball")
clock = pygame.time.Clock()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Ball
ball_radius = 10
ball_x, ball_y = WIDTH // 2, HEIGHT // 2
ball_dx, ball_dy = 4, -4

# Flipper
flipper_length = 100
flipper_angle = 0
flipper_speed = 5
flipper_x, flipper_y = WIDTH // 2, HEIGHT - 50

# Bumpers
bumpers = [(200, 200), (600, 200), (400, 100)]
bumper_radius = 30

def draw_ball():
    pygame.draw.circle(screen, WHITE, (int(ball_x), int(ball_y)), ball_radius)

def draw_flipper():
    end_x = flipper_x + flipper_length * math.cos(math.radians(flipper_angle))
    end_y = flipper_y - flipper_length * math.sin(math.radians(flipper_angle))
    pygame.draw.line(screen, GREEN, (flipper_x, flipper_y), (end_x, end_y), 8)

def draw_bumpers():
    for bx, by in bumpers:
        pygame.draw.circle(screen, RED, (bx, by), bumper_radius)

running = True
while running:
    screen.fill(BLACK)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Controls
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        flipper_angle += flipper_speed
    if keys[pygame.K_RIGHT]:
        flipper_angle -= flipper_speed

    # Ball movement
    ball_x += ball_dx
    ball_y += ball_dy

    # Wall bounce
    if ball_x - ball_radius < 0 or ball_x + ball_radius > WIDTH:
        ball_dx *= -1
    if ball_y - ball_radius < 0:
        ball_dy *= -1

    # Flipper collision
    flipper_end_x = flipper_x + flipper_length * math.cos(math.radians(flipper_angle))
    flipper_end_y = flipper_y - flipper_length * math.sin(math.radians(flipper_angle))
    flipper_rect = pygame.Rect(min(flipper_x, flipper_end_x),
                               min(flipper_y, flipper_end_y),
                               abs(flipper_end_x - flipper_x),
                               abs(flipper_end_y - flipper_y))
    if flipper_rect.collidepoint(ball_x, ball_y):
        ball_dy *= -1
        ball_dx += random.choice([-1, 1])

    # Bumper collisions
    for bx, by in bumpers:
        dist = math.hypot(ball_x - bx, ball_y - by)
        if dist < ball_radius + bumper_radius:
            ball_dx *= -1
            ball_dy *= -1

    draw_ball()
    draw_flipper()
    draw_bumpers()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
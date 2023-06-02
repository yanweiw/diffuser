import sys, os
import numpy as np
import pygame
import random
import torch
import argparse
import diffuser.utils as utils
from diffuser.guides.policies import Policy


def xy2gui(xy):
    # 0 < x < 9
    # 0 < y < 12
    # x = (xy[0] + grid_bd_x) / grid_bd_x * size[0]/2
    x = (xy[0] + 0.5) / 9 * size[0]
    y = (xy[1] + 0.5) / 12 * size[1]
    # y = size[1] - ((xy[1] + grid_bd_y) / grid_bd_y * size[1]/2)
    return np.array([y, x], dtype=float)

# def gui2xy(gui):
#     x = gui[0] / size[0] * 9 - 0.5
#     y = (size[1] - gui[1]) / size[1] * 12 - 0.5
#     # x = gui[0] / size[0] *2*grid_bd - grid_bd
#     # y = (size[1] - gui[1]) / size[1] *2*grid_bd - grid_bd
#     return np.array([x, y, 0, 0], dtype=float)

def infer_target(model, obj_hist, pred_len=64, num_samples=1):
    assert pred_len + history_length <= max_traj_len

    cond = {}
    for t in range(len(obj_hist)):
        cond[t] = obj_hist[t]

    actions, samples = policy(cond, batch_size=num_samples)

    # obj_hist = torch.tensor(np.stack(obj_hist)) # shape (seq_len, 2)
    # tokens = tokenize(obj_hist) # shape (seq_len,)
    # # tokens = tokens.view(1, -1)
    # tokens = tokens.repeat(num_samples, 1) # shape (num_samples, seq_len)
    # assert tokens.shape == (num_samples, len(obj_hist))
    # with torch.no_grad():
    #     out = model.generate(tokens.long().cuda(), pred_len, do_sample=True, temperature=3.0)[:, -pred_len:]
    # out = detokenize(out.cpu().numpy().reshape((1, -1))).reshape((num_samples, pred_len, 2))
    return samples


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--wts', help='weight_path')
    # # parser.add_argument('--gt', action='store_true', help='use ground truth DS controller')
    # parser.add_argument('--cg', action='store_true', help='change goals when perturbation is large')
    # parser.add_argument('--model', default='nano', help='specify model type')
    # args = parser.parse_args()

    max_traj_len = 128
    # Set the maximum speed
    max_attract_speed = 0.5 # use ds policy to tune this; control how fast/how many states visitied
    max_perturb_speed = max_attract_speed
    epsilon = sys.float_info.epsilon

    class Parser(utils.Parser):
        dataset: str = 'maze2d-large-v1'
        config: str = 'config.maze2d'

    args = Parser().parse_args('plan')

    diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

    diffusion = diffusion_experiment.ema
    dataset = diffusion_experiment.dataset
    renderer = diffusion_experiment.renderer

    policy = Policy(diffusion, dataset.normalizer)

    # Initialize Pygame
    pygame.init()
    size = (1000, 1000)
    grid_bd_x = 4.5 #renderer._extent[1] / 2 # 4.5
    grid_bd_y = 6 #renderer._extent[3] / 2 # 6
    background = renderer._background

    # Define the history of the object's positions
    obj_history_xy = [np.array([1,7,0,0])]
    obj_history = [xy2gui(obj_history_xy[0])]
    perturb_history = []
    history_length = max_traj_len / 2
    # Set the colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (150, 150, 150)
    BLUE = (0, 0, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    # Create the screen
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Reach Goal")
    font = pygame.font.Font(None, 36)
    # Create a clock to control the frame rate
    clock = pygame.time.Clock()
    # Add new variables to store the start and end positions of the line and a flag to indicate if the line should be drawn
    line_start_pos = None
    line_end_pos = None

    # if args.wts is not None:
    #     # loading learned model
    #     model_config = GPT.get_default_config()
    #     if args.model == 'nano':
    #         model_config.model_type = 'gpt-nano'
    #     else:
    #         model_config.model_type = 'gpt-micro'
    #     model_config.vocab_size = 100*100
    #     model_config.block_size = max_traj_len-1
    #     model = GPT(model_config)
    #     weight_path = os.path.join('weights', args.wts)
    #     if torch.cuda.is_available():
    #         model.load_state_dict(torch.load(weight_path, map_location=torch.device('cuda:0')))
    #     else:
    #         model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    #     model.set_context_mask(5)
    #     if torch.cuda.is_available():
    #         model.cuda()
    #     model.eval()

    # Run the game loop
    running = True
    while running:
        # Calculate velocity
        # if args.wts is None:
        #     obj_vel = ds.modulate(env.s()[:2]) 
        if False:
            pass
        else:   
            samples = infer_target(policy, obj_history_xy)
            pred_trajs = samples.observations
            # assert pred_trajs.shape[1] > 1 # empirically the immediate prediction gets the agent stuck, so use the next one
            obj_target_pos = pred_trajs[0, len(obj_history_xy), :]
            # obj_target_pos = pred_trajs[:, 1].mean(axis=0)

            # obj_target_pos = infer_target(model, obj_history_xy)
            # assert obj_target_pos.shape == obj_history_xy[-1].shape
        #     obj_vel = (obj_target_pos[:2] - obj_history_xy[-1][:2])

        # obj_speed = np.linalg.norm(obj_vel)
        # if obj_speed > max_attract_speed:
        #     obj_vel *= max_attract_speed / obj_speed
        
        # # Handle events
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         running = False
        #         break
        #     if any(pygame.mouse.get_pressed()):
        #         if line_start_pos is None:
        #             line_start_pos = np.array(pygame.mouse.get_pos())
        #         line_end_pos = np.array(pygame.mouse.get_pos())
        #         perturb_vel = (line_end_pos - line_start_pos).astype(np.float32)
        #         perturb_vel = perturb_vel / size * grid_bd * 2
        #     else:
        #         # reset goal
        #         if args.cg and line_end_pos is not None and line_start_pos is not None and np.linalg.norm((line_end_pos - line_start_pos)) > 300:
        #             ds.set_arena(env.arena, target=gui2xy(line_end_pos))
        #         line_end_pos = None
        #         line_start_pos = None      
        #         perturb_vel = np.array([0.0, 0.0])      
        
        # # Add the perturbation velocity to the object velocity
        # if line_start_pos is not None:
        #     perturb_speed = np.linalg.norm(perturb_vel)
        #     if perturb_speed > max_perturb_speed:
        #         perturb_vel *= max_perturb_speed / perturb_speed
        #     obj_vel[0] = perturb_vel[0] #/ size[0] * grid_bd * 2
        #     obj_vel[1] = -perturb_vel[1] #/ size[1] * grid_bd * 2

        # env step
        # s, _, done, _ = env.step(obj_vel)
        # env.t = 0
        obj_target_pos = obj_history_xy[-1] + 0.2*(obj_target_pos - obj_history_xy[-1])

        obj_history_xy.append(obj_target_pos)
        obj_pos = xy2gui(obj_target_pos)
        # obj_pos[obj_pos<0] = 0
        # if obj_pos[0] > size[0] : obj_pos[0] = size[0]
        # if obj_pos[1] > size[1] : obj_pos[1] < size[1]

        # # plot perturb line
        obj_history.append(obj_pos.copy())
        # if line_start_pos is not None:
        #     perturb_history.append(True)
        # else:
        #     perturb_history.append(False)
        while len(obj_history) > history_length:
            obj_history.pop(0)
            obj_history_xy.pop(0)
            # perturb_history.pop(0)

        # Clear the screen
        # grid_rgb = 255-np.swapaxes(np.repeat(env.arena.occ_grid[::-1, :, np.newaxis] * 255, 3, axis=2).astype(np.uint8), 0, 1)
        surface = pygame.surfarray.make_surface(255-np.swapaxes(np.repeat(background[:, :, np.newaxis] * 255, 3, axis=2).astype(np.uint8),0,1))
        surface = pygame.transform.scale(surface, size)
        screen.blit(surface, (0, 0))

        # # Draw the object and the attractor
        # for target_pos in [[-1, -1], [1, 1], [-1, 1], [1, -1]]:
        #     att = xy2gui(target_pos) 
        #     pygame.draw.circle(screen, GREEN, (int(att[0]), int(att[1])), 10)
        #     pygame.draw.circle(screen, BLUE, (int(obj_pos[0]), int(obj_pos[1])), 5)

        # Draw the history
        for i in range(len(obj_history)):
            # if perturb_history[i]:
                # color = RED
            # else:
                # color = BLUE
            color = BLUE
            pygame.draw.circle(screen, color, (int(obj_history[i][0]), int(obj_history[i][1])), 5)

        # Draw future predictions
        # color = GRAY
        for pred in pred_trajs:
            COLOR = np.random.randint(150, 255, (3,))
            seq = []
            for xy in pred[len(obj_history_xy):]:
                gui_pos = xy2gui(xy)
                seq.append(gui_pos)
                # pygame.draw.circle(screen, COLOR, (int(gui_pos[0]), int(gui_pos[1])), 2)
            pygame.draw.lines(screen, COLOR, False, seq, 3)
                         
        # # Update the display
        # if line_start_pos is not None and line_end_pos is not None:
        #     pygame.draw.line(screen, BLACK, line_start_pos, line_end_pos, 2)

        pygame.display.flip()
        clock.tick(30) # Control the frame rate

    # Quit Pygame
    pygame.quit()

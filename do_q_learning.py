import gym
import numpy as np
from gym import spaces, envs
from envs.simulator import NetworkSimulatorEnv
from agents.q_agent import networkTabularQAgent
from shapely.geometry import LineString
from shapely.geometry import Point
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pylab as pl
####
# This script currently makes the agent use the a random policy to explor the state,action space
# then tests the value of the learning by every 1000 iterations using the best choice and printing the reward
####

def afficher_noeud(coords, color, r, num, ax, plt):
    x, y = coords
    mult = 600
    a = 15
    font_size = 8
    x = float(x) * mult + a
    y = float(y) * mult + a
    drawing = plt.Circle((x, y), r, color=color, fill=True)
    ax.add_artist(drawing)
    plt.text((x - font_size), (y - font_size), str(num), color="white", fontsize=font_size)
    
def afficher_liens(links, coords, color, ax):
    k = 600
    m = 15    
    for i in range(0, len(links)):
        _in = coords[i]
        #print(">>>>> _in: ", _in) 
        for j in range(0, len(links[i])):
            _out = coords[links[i][j]]
            #print(">>>>>> _out: ", _out)
            drawing, = plt.plot([float(_in[0]) * k + m, float(_out[0]) * k + m], [float(_in[1]) * k + m, float(_out[1]) * k + m], color = color)  
            ax.add_artist(drawing)
            
def afficher_shortest_path(coords, path, color, ax): 
    k = 600
    m = 15 
    if path: 
        for i in range(len(path) - 1): 
            _in = coords[path[i]] 
            _out = coords[path[i+1]] 
            drawing, = plt.plot([float(_in[0]) * k + m, float(_out[0]) * k + m], 
                                 [float(_in[1]) * k + m, float(_out[1]) * k + m], 
                                 color=color, linewidth=2) # Red line for shortest path 
            ax.add_artist(drawing) 
    
def afficher(coords, links, path, src, dest, r, namefile, ax):           
    #Afficher les arcs (liaisons) entre chaque noeud
    afficher_liens(links, coords, 'black', ax)
    
    #Afficher le chemin de routage  en orange
    afficher_shortest_path(coords, path, 'orange', ax)
    
    # Afficher tous les neouds en noir
    i = 0
    for node in coords:        
        afficher_noeud(node, 'black', r, i, ax, pl)
        i = i + 1
    
    #Afficher les noeuds intermédiaires du chemin en orange
    for i in range(1, len(path) - 1):
        act_node = path[i]
        afficher_noeud(coords[act_node], 'orange', r, act_node, ax, plt)
    
    # Afficher le noeud source en bleu
    afficher_noeud(coords[src], 'blue', r, src, ax, plt)
        
    # Afficher le noeud de destination en vert
    afficher_noeud(coords[dest], 'green', r, dest, ax, plt)
    
    #Afficher le noeud de l'action en orange
    #afficher_noeud(coords[act_node], 'orange', r, act_node, ax, pl)
    legend_elements = [ 
    plt.Line2D([0], [0], marker='o', color='w', label='Noeud', markerfacecolor='black', markersize=10), 
    plt.Line2D([0], [0], marker='o', color='w', label='Noeud Source', markerfacecolor='blue', markersize=10), 
    plt.Line2D([0], [0], marker='o', color='w', label='Noeud Destination', markerfacecolor='green', markersize=10), 
    plt.Line2D([0], [0], marker='o', color='w', label='Noeud Intermédiaire', markerfacecolor='orange', markersize=10), 
    plt.Line2D([0], [0], color='black', lw=2, label='Lien'),
    plt.Line2D([0], [0], color='orange', lw=2, label='lien actif')] 
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3) #Added Legend 
    
    plt.savefig(f'{namefile}-env simu.png')
    plt.show()
    
def plot_config(namefile, largeur = 500, longueur = 400):
    plt.figure(figsize=(20, 16) ,dpi=600)
    fig, ax = plt.subplots() 
    ax.set_aspect(1)
    L = largeur
    l = longueur
    ax.add_patch(Rectangle((0, 0), L, l, fill=False))
    plt.title(f'Environnement de simulation - {namefile}')
    plt.xlim(0, L)
    plt.ylim(0, l)
    
    return ax

def get_best_action(src, dest, nlinks, q):  
    best_action = 0 
    max_q = q[src][dest][0]  # Initialise avec la première valeur Q 
 
    for action in range(nlinks[src]):
        if q[src][dest][action] < max_q: 
            max_q = q[src][dest][action] 
            best_action = action 
   
    return best_action 

def reconstruct_path(source_node, destination_node, q, links, nlinks): 
    find = True
    path = [source_node] 
    current_node = source_node 
    max_iterations = 1000 # Limite pour éviter les boucles infinies 
    for i in range(max_iterations): #ajout d'une limite pour éviter les boucles infinies 
        if current_node == destination_node: 
            break 
        best_action = get_best_action(current_node, destination_node, nlinks, q) 
        next_node = links[current_node][best_action] 
        path.append(next_node) 
        current_node = next_node
    if current_node != destination_node: 
        print("Aucun chemin trouvé car max itérations")
        find = False
 
    return (find, path) 


def main():
    callmean = 1.0
    for i in range(10):
        callmean += 1.0
        env = NetworkSimulatorEnv()
        state_pair = env.reset()
        env.callmean = callmean
        agent = networkTabularQAgent(env.nnodes, env.nedges, env.distance, env.nlinks)
        done = False
        r_sum_random = r_sum_best = 0
        config = agent.config
        avg_delay = []
        avg_length = []

        for t in range(50001):
            path = []
            if not done:

                current_state = state_pair[1]
                n = current_state[0]
                dest = current_state[1]

                for action in range(env.nlinks[n]):
                    reward, next_state = env.pseudostep(action)
                    agent.learn(current_state, next_state, reward, action, done)

                action  = agent.act(current_state)
                state_pair, reward, done, _ = env.step(action)

                next_state = state_pair[0]
                agent.learn(current_state, next_state, reward, action, done)
                r_sum_random += reward
                
                if env.routed_packets > 0:
                    avg_delay.append(float(env.total_routing_time)/float(env.routed_packets))
                    avg_length.append(float(env.total_hops)/float(env.routed_packets))
                  

                if t%10000 == 0:

                    #if env.routed_packets != 0:
                    #    print ("q learning with callmean:{} time:{}, average delivery time:{}, length of average route:{}, r_sum_random:{}".format(i, t, float(env.total_routing_time)/float(env.routed_packets), float(env.total_hops)/float(env.routed_packets), r_sum_random))


                    current_state = state_pair[1]
                    n = current_state[0]
                    dest = current_state[1]

                    for action in range(env.nlinks[n]):
                        reward, next_state = env.pseudostep(action)
                        agent.learn(current_state, next_state, reward, action, done)

                    action  = agent.act(current_state, True)
                    state_pair, reward, done, _ = env.step(action)

                    next_state = state_pair[0]
                    agent.learn(current_state, next_state, reward, action, done)
                    r_sum_best += reward

                    if env.routed_packets != 0:
                        print("---->src: {}  dest: {}".format(n, dest))
                        a_path, v = env.shortest_path(n, dest)
                        print(">>>>>>> Path Compute best: {}  visited nodes (hops): {}".format(a_path, v))
                        
                        find, path = reconstruct_path(n, dest, agent.q, env.links, env.nlinks)
                        print(">>>>>>>> path: ", path)
                        print ("q learning with callmean:{} time:{}, average delivery time:{}, length of average route:{}, r_sum_best:{}\n\n".format(i, t, float(env.total_routing_time)/float(env.routed_packets), float(env.total_hops)/float(env.routed_packets), r_sum_best))

                        if find:
                            ax = plot_config("Q-Routing")
                            afficher(env.coords, env.links, path, n, dest, 15, "q-routing", ax)
                        else:
                            ax = plot_config("Q-Routing")
                            afficher(env.coords, env.links, a_path, n, dest, 15, "q-routing", ax)
                        path = []
                        
    """#Affichage des graphiques de performance
    plt.figure(figsize=(12, 6))
    plt.plot(avg_delay, label='Delai moyen')
    plt.title('Evolution du delai d\'acheminement des paquets au fil des époques')
    plt.xlabel('Epoques')
    plt.ylabel('Delai (ms)')
    plt.legend()
    plt.savefig('Delai Q-Routing.png')
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(avg_length, label='Nombre de sauts Q-Learning')
    plt.title('Evolution de la longueur de la route des paquets au fil des époques')
    plt.xlabel('Epoques')
    plt.ylabel('Longueur (sauts)')
    plt.legend()
    plt.savefig('Longueur Q-Routing.png')
    plt.show()"""


if __name__ == '__main__':
    main()

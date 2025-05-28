import torch 
import numpy as np 
import random 


def th_min_angle(start, end, radians=False):
    """
    Finds the angle you would add to <start> in order to get to <end>
    on the shortest path.
    """
    a,b,c = (np.pi, 2*np.pi, 3*np.pi) if radians else (180, 360, 540)
    shortest_angle = ((((end - start) % b) + c) % b) - a
    return shortest_angle


def th_interpolate_angles(start, end, T, n_diffuse,mindiff=None, radians=True):
    """
    
    """
    # find the minimum angle to add to get from start to end
    angle_diffs = th_min_angle(start, end, radians=radians)
    if mindiff is not None:
        assert torch.sum(mindiff.flatten()-angle_diffs) == 0.
    if n_diffuse is None:
        # default is to diffuse for max steps 
        n_diffuse = torch.full((len(angle_diffs)), T)


    interps = []
    for i,diff in enumerate(angle_diffs):
        N = int(n_diffuse[i])
        actual_interp = torch.linspace(start[i], start[i]+diff, N)
        whole_interp = torch.full((T,), float(start[i]+diff))
        whole_interp[:N] = actual_interp
        
        interps.append(whole_interp)

    return torch.stack(interps, dim=0)


def get_aa_schedule(T, L, nsteps=100):
    """
    Returns the steps t when each amino acid should be decoded, 
    as well as how many steps that amino acids chi angles will be diffused 
    
    Parameters:
        T (int, required): Total number of steps we are decoding the sequence over 
        
        L (int, required): Length of protein sequence 
        
        nsteps (int, optional): Number of steps over the course of which to decode the amino acids 

    Returns: three items 
        decode_times (list): List of times t when the positions in <decode_order> should be decoded 

        decode_order (list): List of lists, each element containing which positions are going to be decoded at 
                             the corresponding time in <decode_times> 

        idx2diffusion_steps (np.array): Array mapping the index of the residue to how many diffusion steps it will require 

    """
    # nsteps can't be more than T or more than length of protein
    if (nsteps > T) or (nsteps > L):
        nsteps = min([T,L])

    
    decode_order = [[a] for a in range(L)]
    random.shuffle(decode_order)
    
    while len(decode_order) > nsteps:
        # pop an element and then add those positions randomly to some other step
        tmp_seqpos = decode_order.pop()
        decode_order[random.randint(0,len(decode_order)-1)] += tmp_seqpos
        random.shuffle(decode_order)
    
    decode_times = np.arange(nsteps)+1
    
    # now given decode times, calculate number of diffusion steps each position gets
    aa_masks = np.full((200,L), False)
    
    idx2diffusion_steps = np.full((L,),float(np.nan))
    for i,t in enumerate(decode_times):
        decode_pos = decode_order[i]    # positions to be decoded at this step 
        
        for j,pos in enumerate(decode_pos):
            # calculate number of diffusion steps this residue gets 
            idx2diffusion_steps[pos] = int(t)
            aa_masks[t,pos] = True
    
    aa_masks = np.cumsum(aa_masks, axis=0)
    
    return decode_times, decode_order, idx2diffusion_steps, ~(aa_masks.astype(bool))

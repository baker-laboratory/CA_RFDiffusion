import torch
import numpy as np
import random
import glob
import os

import rf_diffusion.conditions.ss_adj.sec_struct_adjacency as sec_struct_adj
from rf_diffusion.conditions.ss_adj.sec_struct_adjacency import SS_LOOP, SS_MASK, ADJ_MASK
from rf_diffusion.contigs import ContigMap

class ScaffoldLoader:
    '''
    Base class for scaffold loading
    '''

    def __init__(self, conf):
        """
        Args:
            conf (OmegaConf): Omegaconf for inference time
        """    
        self.conf = conf


    def __getitem__(self, idx):
        '''
        The base class returns outputs that do not change anything

        Returns:
            contig_conf (OmegaConf): The config to use for ContigMap generation
            ss_adj (SecStructAdjacency): The secondary structure adjacency object for this scaffold
        '''
        return self.conf.contigmap, sec_struct_adj.SecStructAdjacency()

    def __len__(self):
        return 1

    def set_target_feats(self, target_feats):
        self.target_feats = target_feats
    

class ScaffoldFirstLoader(ScaffoldLoader):
    '''
    A base class to make further ScaffoldLoaders easier to write
    This class assumes that the scaffold comes before the non-diffused portion and adjusts things accordingly
    '''

    def get_scaffold(self, idx):
        '''
        Get the scaffold corresponding to idx

        Args:
            idx (int): scaffold index

        Returns:
            contig_conf_overrides (dict): Which keys of conf.contigmap do you want to overwrite?
            ss (torch.Tensor[long]): The SecondaryStructure tensor for the scaffold [binderlen or L]
            adj (torch.Tensor[long]): The AdjacencyMatrix tensor for the scaffold [binderlen or L,binderlen or L]
            scaff_deterministic_L (bool): Is the length of the ss deterministic based on the input config?

        '''
        assert False, 'ScaffoldFirstLoader get_scaffold() called. Derived classes must implement this method.'

    def __len__(self):
        assert False, 'ScaffoldFirstLoader __len__() called. Derived classes must implement this method.'


    def __getitem__(self, idx):
        '''
        Calls derived self.get_scaffold() and integrates the ss, adj, and contig into the current situation

        Returns:
            contig_conf (OmegaConf): The config to use for ContigMap generation
            ss_adj (SecStructAdjacency): The secondary structure adjacency object for this scaffold
        '''

        contig_conf_overrides, ss, adj, scaff_deterministic_L = self.get_scaffold(idx)
        binderlen = len(ss)

        # Apply contig conf overrides
        contig_conf = self.conf.contigmap.copy()
        for key, value in contig_conf_overrides.items():
            assert hasattr(contig_conf, key)
            setattr(contig_conf, key, value)

        contig_map = ContigMap(self.target_feats, **contig_conf)
        mapping = contig_map.get_mappings()
        entirely_diffused = (~mapping['inpaint_str']).all()

        assert (~mapping['inpaint_str']).any(), 'Cannot generate scaffold because nothing is diffused!'

        # There are 4 scenarios
        # 1. monomer design, entire ss/adj specified, there might be motifs
        # 2. binder design, entire ss/adj specified

        # 3. binder design w/motifs, only binder specified
        # 4. binder design w/o motifs, only binder specified

        # Case 1 or Case 2

        # The size of the scaffold precisely matches the size of the indep
        if binderlen == len(contig_map.inpaint_str) and scaff_deterministic_L and contig_map.deterministic:
            print("SS/ADJ: Specified exactly for entire input stucture (because contig length matched ss/adj length")
            ss_adj = sec_struct_adj.SecStructAdjacency(ss=ss, adj=adj)
            return contig_conf, ss_adj

        # Additional case 1 (entirely-specified monomer design) but the user specified the wrong size for the contig
        if entirely_diffused:
            ss_adj = sec_struct_adj.SecStructAdjacency(ss=ss, adj=adj)
            print(f'SS/ADJ: Assuming monomer design. Overriding protein length with length from ss/adj: {binderlen}')
            contig_conf.contigs = [f'{binderlen}']
            assert ContigMap(self.target_feats, **contig_conf).deterministic, 'After modifying contigmap.contigs, your contig is still non-deterministic which is incompatible with scaffolds'
            return contig_conf, ss_adj
           

        # There are diffused and non-diffused portions and the entire ss/adj matrix has not been specified
        #   This implies that they are doing binder design

        # Next, let's check if the pdbs is set up such that all of the diffused residues
        #  come before all of the non-diffused residues. Very clearly binder design w/o motifs

        # Case 3. binder design w/motifs, only binder specified

        inpainted_seq_regions = sec_struct_adj.repeating_regions(mapping['inpaint_str'])
        if len(inpainted_seq_regions) == 2 and not inpainted_seq_regions[0][0]:

            binder_part = [f'{binderlen},0']
            new_contig = ['_'.join(binder_part+contig_map.sampled_mask[1:])]
            contig_conf.contigs = new_contig

            print(f'SS/ADJ: Assuming binder design. Overriding binder length with length from ss/adj: {binderlen}')

            contig_map = ContigMap(self.target_feats, **contig_conf)
            assert contig_map.deterministic, 'After modifying contigmap.contigs, your contig is still non-deterministic which is incompatible with scaffolds'

            ss_adj = self.store_ss_adj_at_front(ss, adj, len(contig_map.inpaint_str))

            return contig_conf, ss_adj


        # At this point, either the user has specified something incorrectly or we're dealing with a motif grafting scenario
        # Allowing variable-length sections in-between motifs is theoretically possible if the ss/adj generation code is reworked
        #  however, for now, it'll cause glitches, so we enforce exact-length

        assert scaff_deterministic_L and contig_map.deterministic, ("You've specified an unsupported ss/adj / config combo. Here are some mistakes"
            " of what you might have committed: Target comes first in contig. Motif grafting with variable-sized diffused regions.")


        # Figure out which chains are present within the span covered by binderlen and which chains are present outside that range
        chains_in_ss = set()
        chains_in_else = set()

        for i in range(binderlen):
            chains_in_ss.add(contig_map.hal[i][0])

        for i in range(binderlen, len(contig_map.hal)):
            chains_in_else.add(contig_map.hal[i][0])

        assert len(chains_in_ss & chains_in_else) == 0, ("We've deduced you're trying to do binder design with motifs, only specifing SS/ADJ for the binder."
                    " But your SS/ADJ share chains with the target which is almost certainly a length error somewhere.")

        # Case 4
        ss_adj = self.store_ss_adj_at_front(ss, adj, len(contig_map.inpaint_str))

        print('SS/ADJ: Assuming motif-graft binder design.')

        return contig_conf, ss_adj


    def store_ss_adj_at_front(self, ss, adj, L):
        binderlen = len(ss)

        ss_adj = sec_struct_adj.SecStructAdjacency(full_mask_length=L)
        ss_adj.ss[:binderlen] = ss
        ss_adj.adj[:binderlen,:binderlen] = adj

        return ss_adj


class FileSSADJScaffoldLoader(ScaffoldFirstLoader):
    '''
    A scaffold loader to load ss/adj files like the original Joe + Nate diffusion

    Expects one of:
        conf.scaffoldguided.scaffold_dir: A directory where the <scaffname>_ss.pt and <scaffname>_adj.pt files are stored
        conf.scaffoldguided.scaffold_arc: A torch.pt dictionary that maps scaffname to tuple(ss, adj)

    Where the stored:
        ss is torch.Tensor[long] [L]
        adj is torch.Tensor[long] [L,L]

    One can also specify conf.scaffoldguided.scaffold_list to be either a list or a path to a .txt file
        list: Must be a list of strings that correspond to scaffold names
        .txt file: A list of the scaffold names to use. May additionally contain a contig for each scaffold after the name

        --> If conf.scaffoldguided.scaffold_list is not specified, all scaffolds in the scaffold_dir or scaffold_arc are used

    Either loads ss/adj for the binder or for the whole pdb
    '''
    def __init__(self, conf):
        """
        Args:
            conf (OmegaConf): Omegaconf for inference time
        """    
        super().__init__(conf)
        sg_conf = conf.scaffoldguided
        self.sg_conf = sg_conf

        # path to directory with scaffolds, ss files and block_adjacency files
        self.scaffold_dir = sg_conf.scaffold_dir

        # Alternatively, you can pass a torch.pt dictionary with <scaffold>_ss, <scaffold>_adj
        self.scaffold_arc = None
        if sg_conf.scaffold_arc:
            self.scaffold_arc = torch.load(sg_conf.scaffold_arc, weights_only=False)

        assert self.scaffold_dir or self.scaffold_arc, 'Where are your scaffolds? Either pass scaffoldguided.scaffold_dir of scaffoldguided.scaffold_arc'
        assert bool(self.scaffold_dir) ^ bool(self.scaffold_arc), 'Pick only one of: scaffoldguided.scaffold_dir and scaffoldguided.scaffold_arc'

        # either list or path to .txt file with list of scaffolds
        self.contig_list = []
        if sg_conf.scaffold_list is not None:
            if isinstance(sg_conf.scaffold_list, list):
                self.scaffold_list = scaffold_list
            elif sg_conf.scaffold_list[-4:] == '.txt':
                # Load .txt file and parse out scaffold names and potential contigs
                list_from_file = []
                with open(sg_conf.scaffold_list,'r') as f:
                    for line in f:
                        line = line.strip()
                        if len(line) == 0: continue
                        sp = line.split()
                        list_from_file.append(sp[0])
                        if len(sp) > 1:
                            self.contig_list.append(sp[1])
                self.scaffold_list = list_from_file
            else:
                raise NotImplementedError
        else:
            if self.scaffold_dir:
                # This is bad but this is what Nate and Joe did. Run ls in the scaffold_dir folder
                self.scaffold_list = [os.path.basename(x)[:-len('_ss.pt')] for x in glob.glob(f'{self.scaffold_dir}/*_ss.pt')]
                if len(self.scaffold_list) > 100:
                    print(f'Warning! Running ls in {self.scaffold_dir}. This may cause I/O issues. Use scaffold_arc or scaffold_list to prevent this.')
            else:
                self.scaffold_list = [x[:-len('_ss')] for x in list(self.scaffold_arc) if x.endswith('_ss')]

        # Make sure the contig_list length matches the number of scaffolds
        if len(self.contig_list) > 0:
            assert len(self.contig_list) == len(self.scaffold_list), "If you're going to specify contigs in the scaffold_list, every scaffold must have a contig"
        else:
            self.contig_list = None

        # the old code used to print all of them...
        print(f'Found {len(self.scaffold_list)} scaffolds. First {min(10, len(self.scaffold_list))}')
        print('\n'.join([f'   {x}' for x in self.scaffold_list[:10]]))


        # maximum sampled insertion in each loop segment
        if '-' in str(sg_conf.sampled_insertion):
            self.sampled_insertion = [int(str(sg_conf.sampled_insertion).split("-")[0]), int(str(sg_conf.sampled_insertion).split("-")[1])]
        else:
            self.sampled_insertion = [0, int(sg_conf.sampled_insertion)]        

        # maximum sampled insertion at N- and C-terminus
        if '-' in str(sg_conf.sampled_N):
            self.sampled_N = [int(str(sg_conf.sampled_N).split("-")[0]), int(str(sg_conf.sampled_N).split("-")[1])]
        else:
            self.sampled_N = [0, int(sg_conf.sampled_N)]
        if '-' in str(sg_conf.sampled_C):
            self.sampled_C = [int(str(sg_conf.sampled_C).split("-")[0]), int(str(sg_conf.sampled_C).split("-")[1])]
        else:
            self.sampled_C = [0, int(sg_conf.sampled_C)]

        # number of residues to mask ss identity of in H/E regions (from junction)
        # e.g. if ss_mask = 2, L,L,L,H,H,H,H,H,H,H,L,L,E,E,E,E,E,E,L,L,L,L,L,L would become\
        # M,M,M,M,M,H,H,H,M,M,M,M,M,M,E,E,M,M,M,M,M,M,M,M where M is mask
        self.ss_mask = sg_conf.ss_mask

        # whether or not to work systematically through the list
        if not sg_conf.systematic:
            # implement this by shuffling the list instead of randomint like the original
            new_idx = np.arange(len(self.scaffold_list))
            np.random.shuffle(new_idx)
            self.scaffold_list = list(np.array(self.scaffold_list, dtype=object)[new_idx])
            if self.contig_list:
                self.contig_list = list(np.array(self.contig_list, dtype=object)[new_idx])



        self.systematic = sg_conf.systematic

        # Previously all loops were set to ADJ_FAR which is not ideal
        # If this is detected, set them to ADJ_MASK
        # Use this flag in the rare circumstance that you really want all loops set to ADJ_FAR
        self.not_legacy_adj = sg_conf.not_legacy_adj

        # whether to mask loops or not
        self.mask_loops = sg_conf.mask_loops


    def __len__(self):
        return len(self.scaffold_list)


    def get_scaffold(self, idx):
        '''
        Get the scaffold corresponding to idx

        Args:
            idx (int): scaffold index

        Returns:
            contig_conf_overrides (dict): Which keys of conf.contigmap do you want to overwrite?
            ss (torch.Tensor[long]): The SecondaryStructure tensor for the scaffold [binderlen or L]
            adj (torch.Tensor[long]): The AdjacencyMatrix tensor for the scaffold [binderlen or L,binderlen or L]
            scaff_deterministic_L (bool): Is the length of the ss deterministic based on the input config?

        '''
        assert idx < len(self.scaffold_list)

        item = self.scaffold_list[idx]
        extra_str = ''
        contig_conf_overrides = {}
        if self.contig_list:
            contig_conf_overrides['contigs'] = [self.contig_list[idx]]
            extra_str = f' with contig override: {self.contig_list[idx]}'

        print(f'Scaffold constrained based on file: {item}{extra_str}')
        # load files
        ss, adj = self.get_ss_adj(item)

        if not self.not_legacy_adj:
            ss, adj = sec_struct_adj.convert_legacy_ss_adj(ss, adj)

        # separate into segments (loop or not)
        # mask means: is_loop
        mask = (ss == SS_LOOP) | (ss == SS_MASK)
        # old code expects type, size
        segments = [(tp, end-start+1) for tp,start,end in sec_struct_adj.ss_to_segments(ss)]

        # insert into loops to generate new mask
        # expanded_mask means: wasn't loop in original ss
        expanded_mask = self.expand_mask(mask, segments)

        # expand ss and adj
        ss, adj = self.expand_ss(ss, adj, mask, expanded_mask)

        # finally, mask some proportion of the ss at either end of the non-loop ss blocks
        ss, adj = self.mask_ss_adj(ss, adj, expanded_mask)

        return contig_conf_overrides, ss, adj, self.ss_adj_length_is_deterministic()


    def get_ss_adj(self, item):
        """
        Given at item, get the ss tensor and block adjacency matrix for that item

        Args:
            item (str): Scaffold name

        Returns:
            ss (torch.Tensor[long]): Secondary structure assignment [binderlen or L]
            adj (torch.Tensor[long]): Secondary structure adjacency [binderlen or L, binderlen or L]
        """
        if self.scaffold_dir:
            ss = torch.load(os.path.join(self.scaffold_dir, f'{item.split(".")[0]}_ss.pt'), weights_only=False)
            adj = torch.load(os.path.join(self.scaffold_dir, f'{item.split(".")[0]}_adj.pt'), weights_only=False)
        else:
            tag_part = item.split(".")[0]
            ss = self.scaffold_arc[tag_part + "_ss"]
            adj = self.scaffold_arc[tag_part + "_adj"]

        return ss.long().clone(), adj.long().clone()

    def expand_mask(self, mask, segments):
        """
        Function to generate a new mask with dilated loops and N and C terminal additions

        Args:
            mask (torch.Tensor[bool]): Which parts of the SS are loops or mask? [binderlen or L]
            segments (list[tuple[int,int]]): List of (ss_type, size) for each secondary structural segment of ss

        Returns:
            expanded_mask (list[bool]): A mask of what is new or loop (False) and what comes from the ss (True)
        """
        N_add = random.randint(self.sampled_N[0], self.sampled_N[1])
        C_add = random.randint(self.sampled_C[0], self.sampled_C[1])

        output = N_add * [False]
        for ss, length in segments:
            if ss != SS_LOOP and ss != SS_MASK:
                output.extend(length*[True])
            else:
                # randomly sample insertion length
                ins = random.randint(self.sampled_insertion[0], self.sampled_insertion[1])
                output.extend((length + ins)*[False])
        output.extend(C_add*[False])
        assert torch.sum(torch.tensor(output)) == torch.sum(~mask)
        return torch.tensor(output)

    def expand_ss(self, ss, adj, mask, expanded_mask):
        """
        Actually expand out the ss and adj in the expanded regions storing SS_MASK or ADJ_MASK to the new areas

        Args:
            ss (torch.Tensor[long]): Secondary structure assignment [binderlen or L]
            adj (torch.Tensor[long]): Secondary structure adjacency [binderlen or L, binderlen or L]
            mask (torch.Tensor[bool]): Which parts of the SS are loops or mask? [binderlen or L]
            expanded_mask (list[bool]): A mask of what is new (False) and what comes from the ss (True)

        Returns:
            ss_out (torch.Tensor[long]): Secondary structure assignment but with expansions set to SS_MASK [binderlen or L]
            adj_out (torch.Tensor[long]): Secondary structure adjacency but with expanded areas set to ADJ_MASK [binderlen or L, binderlen or L]

        """
        ss_out = torch.full((expanded_mask.shape[0],), SS_MASK)
        adj_out = torch.full((expanded_mask.shape[0], expanded_mask.shape[0]), ADJ_MASK)
        ss_out[expanded_mask] = ss[~mask]
        expanded_mask_2d = torch.full(adj_out.shape, True)
        #mask out loops/insertions, which is ~expanded_mask
        expanded_mask_2d[~expanded_mask, :] = False
        expanded_mask_2d[:,~expanded_mask] = False

        mask_2d = torch.full(adj.shape, True)
        # mask out loops. This mask is True=loop
        mask_2d[mask, :] = False
        mask_2d[:,mask] = False
        adj_out[expanded_mask_2d] = adj[mask_2d]
        adj_out = adj_out.reshape((expanded_mask.shape[0], expanded_mask.shape[0]))

        return ss_out, adj_out

    def mask_ss_adj(self, ss, adj, expanded_mask):
        """
        Expand loop or mask regions by self.ss_mask into non-loop/non-mask regions using SS_MASK and ADJ_MASK

        Also set SS_LOOP to SS_MASK (and corresponding adj) if self.mask_loops is true

        Args:
            ss (torch.Tensor[long]): Secondary structure assignment [binderlen or L]
            adj (torch.Tensor[long]): Secondary structure adjacency [binderlen or L, binderlen or L]
            expanded_mask (list[bool]): A mask of what is new (False) and what comes from the ss (True)

        Returns:
            ss (torch.Tensor[long]): Secondary structure assignment but with bigger loops [binderlen or L]
            adj (torch.Tensor[long]): Secondary structure adjacency but with bigger loops [binderlen or L, binderlen or L]

        """
        original_mask = torch.clone(expanded_mask)
        if self.ss_mask > 0:
            # This basically shrinks the runs of 1s by self.ss_mask from both sides
            for i in range(1, self.ss_mask+1):
                expanded_mask[i:] *= original_mask[:-i]
                expanded_mask[:-i] *= original_mask[i:]

        ss[~expanded_mask] = SS_MASK

        if self.mask_loops:
            is_loop = ss == SS_LOOP
            ss[is_loop] = SS_MASK
            adj[is_loop,:] = ADJ_MASK
            adj[:,is_loop] = ADJ_MASK

        # mask adjacency
        adj[~expanded_mask] = ADJ_MASK
        adj[:,~expanded_mask] = ADJ_MASK

        return ss, adj

    def ss_adj_length_is_deterministic(self):
        '''
        Determines whether or not this FileSSADJScaffoldLoader returns deterministic length ss/adj files
        '''
        if self.sampled_insertion[1] != self.sampled_insertion[0]:
            return False
        if self.sampled_N[1] != self.sampled_N[0]:
            return False
        if self.sampled_C[1] != self.sampled_C[0]:
            return False
        return True


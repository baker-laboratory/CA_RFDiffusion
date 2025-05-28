from rf_diffusion.train_data.exceptions import NextExampleException


def reject_chain0_longer_than(sel_item=None, max_length=None, verbose=False, **kwargs):
    '''
    A fast dataset filter that rejects examples where chain 0 is longer than a specified length

    Args:
        sel_item (dict): The sel_item from rf2aa dataloader
        max_length (int): The maximum length that chain A can be
    '''

    assert sel_item is not None
    assert max_length is not None

    if 'LEN' in sel_item and isinstance(sel_item['LEN'], list) and len(sel_item['LEN']) > 1:
        if sel_item['LEN'][0] > max_length:
            raise NextExampleException(f'fast_filter: reject_chain0_longer_than: {sel_item["LEN"][0]} > {max_length}', quiet=not verbose)
        else:
            return

    if 'LEN_EXIST' in sel_item:
        if sel_item['LEN_EXIST'] > max_length:
            raise NextExampleException(f'fast_filter: reject_chain0_longer_than: {sel_item["LEN_EXIST"]} > {max_length}', quiet=not verbose)
        else:
            return

    assert False, list(sel_item)








class NextExampleException(Exception):
    '''
    Used in the dataloader to signify that another attempt should be made with the same dataset

    Someday I hope to be able to allow these to be thrown from mask_generator with the current mask
        as a field and then that same mask gets another attempt with a different example
    '''

    def __init__(self, message, quiet=False):
        '''
        Args:
            message: The message of why NextExampleException was called
            quiet: Should this message be shown in the log?
        '''
        self.message = message
        self.quiet = quiet

    def get_message(self):
        '''
        Get the message to be displayed or None if a message should not be displayed
        '''
        if self.quiet:
            return None
        return f'NextExampleException: {self.message}'


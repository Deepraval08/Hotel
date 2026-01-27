import sys

def error_message_detail(error, error_detail: sys):
    _, _, tb = error_detail.exc_info()
    file_name = tb.tb_frame.f_code.co_filename
    line_number = tb.tb_lineno

    return f"Error in {file_name} at line {line_number}: {str(error)}"

class CustomException(Exception):
    def __init__(self, error, error_detail=sys):
        super().__init__(error)
        self.error_message = error_message_detail(error, error_detail)

    def __str__(self):
        return self.error_message

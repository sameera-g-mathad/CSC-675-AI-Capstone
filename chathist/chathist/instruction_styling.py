from abc import ABC, abstractmethod
import pandas as pd
import chathist


class Instruction(ABC):
    """
    Class for creating instruction styles either alpaca or phi3.
    """

    def __init__(self):
        """
        This is the superclass for Alpaca and Phi3 classes that are two
        popular styles for creating prompt styles for finetuning a LLM.

        """
        self._prompt = chathist.config.prompt
        self._input_query = chathist.config.input_query
        self._response_query = chathist.config.response_query
        self._input_col = chathist.config.input_col
        self._response_col = chathist.config.response_col
        self._output_col = chathist.config.output_col
        self._new_df = chathist.config.new_df

    def is_format(self, _input: str) -> bool:
        """Experimental"""
        return (
            self._prompt in _input
            and self._input_query in _input
            and self._response_query in _input
        )

    @abstractmethod
    def format(self, _input: str) -> str:
        """
        Abstract method to style the input that can be used for instruction finetuning the LLM

        :examples:

        `Alpaca`
        >>> prompt = "You are a history bot and given the input,
        you task is to predict a response."
        >>> inpout_query = "Input:"
        >>> response_query = "Response:"
        >>> Output:
          "'### You are a history bot and given the input,'
          'you task is to predict a response.'
          '###Input:'
          'Training input.'
          '###Response:'
          'Training Response'"

        `Phi3`
        >>> inpout_query = "<|user|>"
        >>> response_query = "<|assistant|>"
        >>> Output:
          "'<|user|>'
          'Training input.'
          '<|assistant|>'
          'Training Response'"
        """
        raise NotImplementedError("Needs to be implemented by subclasses")

    def _return_df(
        self,
        df: pd.DataFrame,
        combined_series: pd.Series,
    ) -> pd.DataFrame:
        """
        Method to return new dataframe or modify existing dataframe.
        This method is internally used by convert_train and convert_test,
        and is used as a private method.

        :param pd.DataFrame df: Pandas Dataframe that needs to be modified if opted.
        :param pd.Series combined_series: Pandas Series that contains the
        instructions after style formatting.
        :param str output_col: New column to add to the existing dataframe or to the new dataframe.
        :param bool new_df: Boolean value, that tells whether a new dataframe.
        needs to be returned or not.

        :rtype: pd.DataFrame.
        :returns: A modified or new dataframe.
        """
        if not self._new_df:
            df[self._output_col] = combined_series
            return df

        return pd.DataFrame({self._output_col: combined_series})

    def convert_train(
        self,
        df: pd.DataFrame,
    ):
        """
        This method is used to take input and response columns and return
        a new or existing dataframe with the instruction column added
        that represents the instruction data needed to instruction finetune
        a LLM.

        :param pd.DataFrame df: Dataframe consisting of input and output columns.
        :param input_col str: Column that will be used as the input query to train the LLM.
        :param response_col str: Column that will be used as the response query to train the LLM.
        :param output_col str: New column that will be either added to the existing dataframe or
        to a new dataframe.
        :param bool new_df: Boolean value, that tells whether a new dataframe.
        needs to be returned or not.

        :rtype: pd.DataFrame.
        :returns: A modified or new dataframe.
        """
        combined_series = df.apply(
            lambda row: self.format(row[self._input_col]) + row[self._response_col],
            axis=1,
        )

        return self._return_df(df=df, combined_series=combined_series)

    def convert_test(self, df: pd.DataFrame):
        """
        This method is used to take input column and return
        a new or existing dataframe with the instruction column added
        that represents the instruction data needed to instruction finetune
        a LLM. As the name suggests this method should be used for test data,
        i.e., it will not add response and acts as the `format` method itself,
        but is a workaround for dataframes.

        :param pd.DataFrame df: Dataframe consisting of input and output columns.
        :param input_col str: Column that will be used as the input query to train the LLM.
        :param output_col str: New column that will be either added to the existing dataframe or
        to a new dataframe.
        :param bool new_df: Boolean value, that tells whether a new dataframe.
        needs to be returned or not.

        :rtype: pd.DataFrame.
        :returns: A modified or new dataframe.
        """
        combined_series = df.apply(
            lambda row: self.format(row[self._input_col]), axis=1
        )

        return self._return_df(df=df, combined_series=combined_series)


class Alpaca(Instruction):
    """
    A subclass inheriting Intruction class and implements format method
    that is used to design inputs into popular ***alpaca*** style.
    """

    def format(self, _input: str):
        return (
            f"{self._prompt}" f"{self._input_query}{_input}" f"{self._response_query}"
        )


class Phi3(Instruction):
    """
    A subclass inheriting Intruction class and implements format method
    that is used to design inputs into popular ***phi*** style.
    """

    def format(self, _input: str):
        return f"{self._input_query}{_input}{self._response_query}"


class InstructionStyle:
    """
    Factory for creating instruction style classes be it Alapaca or Phi3
    """

    @staticmethod
    def load() -> Instruction:
        """
        Factory Method for creating an instance of Instruction object that can be use
        to create either ***alpaca*** or ***phi3*** styled prompts.
        :param str style: Type of Instruction style to be instantiated.
        `alpaca` and `phi3` supported now. `alpaca` returned by default.
        :param str prompt: This attribute is used in case of **alpaca** styling
        and can be ignored or passed '' if **phi3** is used.
        :param str input_query: The input query that needs to instruct LLM what the input
        is.
        :param str response_query: The response query hat needs to instruct LLM what
        the response is.

        :rtype: Instruction
        :returns: An Instruction object which is an instance of either `Alpaca` subclass
        or `Phi3` subclass.
        """
        style = chathist.config.style_name
        chathist.config.log.info("%s style chosen!!", style)
        match (style):
            case "alpaca":
                return Alpaca()
            case "phi3":
                return Phi3()

        chathist.config.log.warning(
            "No style chosen!!! Returning default style: Alpaca"
        )
        return Alpaca()

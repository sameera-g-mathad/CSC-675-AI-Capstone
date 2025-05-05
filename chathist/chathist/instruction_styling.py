from abc import ABC, abstractmethod
import pandas as pd
import chathist


class Instruction(ABC):
    """
    Class for creating instruction styles either alpaca or phi3.
    """

    def __init__(self, prompt: str, input_query: str, response_query: str):
        """
        This is the superclass for Alpaca and Phi3 classes that are two
        popular styles for creating prompt styles for finetuning a LLM.

        :param str prompt: The prompt that is mainly used in Alpaca finetuning.
        No need of passing this argument while opting `Phi3` style. Although empty
        '' can be passed as well.
        :param str input_query: The input query that needs to instruct LLM what the input
        is.
        :param str response_query: The response query hat needs to instruct LLM what
        the response is.
        """
        self.prompt = prompt
        self.input_query = input_query
        self.response_query = response_query

        # this is used in dataloader to mask inputs if user chooses to do so.
        chathist.config._set_response_query(response_query=response_query)

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
        output_col: str,
        new_df: bool,
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
        if not new_df:
            df[output_col] = combined_series
            return df

        return pd.DataFrame({output_col: combined_series})

    def convert_train(
        self,
        df: pd.DataFrame,
        input_col: str,
        response_col: str,
        output_col: str,
        new_df: bool = True,
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
            lambda row: self.format(row[input_col]) + row[response_col], axis=1
        )

        return self._return_df(
            df=df, combined_series=combined_series, output_col=output_col, new_df=new_df
        )

    def convert_test(
        self, df: pd.DataFrame, input_col: str, output_col: str, new_df: bool = True
    ):
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
        combined_series = df.apply(lambda row: self.format(row[input_col]), axis=1)

        return self._return_df(
            df=df, combined_series=combined_series, output_col=output_col, new_df=new_df
        )


class Alpaca(Instruction):
    """
    A subclass inheriting Intruction class and implements format method
    that is used to design inputs into popular ***alpaca*** style.
    """

    def format(self, _input: str):
        return f"{self.prompt}" f"{self.input_query}{_input}" f"{self.response_query}"


class Phi3(Instruction):
    """
    A subclass inheriting Intruction class and implements format method
    that is used to design inputs into popular ***phi*** style.
    """

    def format(self, _input: str):
        return f"{self.input_query}{_input}{self.response_query}"


class InstructionStyle:
    """
    Factory for creating instruction style classes be it Alapaca or Phi3
    """

    @staticmethod
    def load(
        style: str = "alpaca",
        prompt: str = "",
        input_query: str = "",
        response_query: str = "",
    ) -> Instruction:
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
        match (style):
            case "alpaca":
                return Alpaca(
                    prompt=prompt,
                    input_query=input_query,
                    response_query=response_query,
                )
            case "phi3":
                return Phi3(
                    prompt="",
                    input_query=input_query,
                    response_query=response_query,
                )

        return Alpaca(
            prompt=prompt,
            input_query=input_query,
            response_query=response_query,
        )

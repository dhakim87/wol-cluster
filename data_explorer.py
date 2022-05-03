import tkinter as tk
from tkinter import filedialog
import cmd
import os
import traceback

from table_info import BiomTable, CSVTable


def _import_woltka_metadata():
    metadata_table = CSVTable("./woltka_metadata.tsv", delimiter="\t")
    metadata_table = metadata_table.load_dataframe()
    return metadata_table


WOLTKA_METADATA = _import_woltka_metadata()


class RootCmd(cmd.Cmd):
    prompt = '(Explorer) '
    intro = 'Welcome to subspace clustering shell.  Type help or ? to list commands.\n"'

    def do_quit(self, args):
        'Exit the command shell'
        return True

    def do_exit(self, args):
        'Exit the command shell'
        return True

    def do_ls(self, args):
        os.system("ls")

    def do_pwd(self, args):
        os.system("pwd")

    def do_open(self, args):
        'Open a Woltka .biom table (table must use "none" taxonomic assignment level).\n OPEN <file.biom>'
        try:
            bt = BiomTable(args)
            df = bt.load_dataframe()
            subshell = BiomCmd(bt, df)
            subshell.cmdloop()
        except Exception as e:
            traceback.print_exc()


class BiomCmd(cmd.Cmd):
    def __init__(self, biom_table, dataframe):
        super().__init__()
        self.prompt="(" + biom_table.biom_filepath + ") "
        self.intro = 'Clustering ' + biom_table.biom_filepath + ' Type help or ? to list commands.\n"'
        self.biom = biom_table
        self.df = dataframe

    def do_close(self, args):
        'Close the biom table and return to the main explorer'
        return True

    def do_list_genera(self, args):
        'Show available for clustering'
        # Should this show all woltka genera or only those present in the loaded dataset?
        genera = WOLTKA_METADATA['genus']
        genera = genera.unique().astype(str).tolist()
        print(sorted(genera))

    def do_genus(self, args):
        'View genus'
        woltka_meta = WOLTKA_METADATA[WOLTKA_METADATA['genus'] == args]
        woltka_meta = woltka_meta[['#genome', 'genus', 'species']]
        print(woltka_meta)
        pass


class GenusCmd(cmd.Cmd):
    def __init__(self, genus, woltka_meta, df):
        super().__init__()
        self.prompt="(" + genus + ") "
        self.intro = 'Type help or ? to list commands.\n"'
        self.woltka_meta = woltka_meta
        self.df = df

    # def do_list


if __name__ == "__main__":
    RootCmd().cmdloop()

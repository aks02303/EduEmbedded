import sys
from pykg2vec.config.config import Importer, KGEArgParser
from pykg2vec.utils.trainer import Trainer
def main():
    # getting the customized configurations from the command-line arguments.
    args = KGEArgParser().get_args(sys.argv[1:])
    # Extracting the corresponding model config and definition from Importer().
    config_def, model_def = Importer().import_model_config(args.model_name.lower())
    config = config_def(args)
    #model = model_def(**config.__dict__)
    model = model_def(config)
    # Create the model and load the trained weights.
    trainer = Trainer(model, config)
    trainer.build_model()

    #trainer.infer_tails(1, 10, topk=5)
    #trainer.infer_heads(10, 20, topk=5)
    #trainer.infer_rels(1, 20, topk=5)

    #head: class_10_chapter_no_6
    #relation: chapter_sequence
    #tail: start_chapter_seq
    #trainer.infer_tails(873, 8, topk=5)
    #trainer.infer_heads(8, 1035, topk=5)
    #trainer.infer_rels(873, 1035, topk=5)

    #head: class_5_chapter_no_13
    #relation: chapter_sequence
    #tail: chapter_no_12
    trainer.infer_tails(0, 1, topk=5)
    trainer.infer_heads(1, 890, topk=5)
    trainer.infer_rels(0, 890, topk=5)


if __name__ == "__main__":
    main()


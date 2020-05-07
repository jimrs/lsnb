from LooselySymmetricNB import LooselySymmetricNB
import utils.preprocessing


iris_path = "datasets/iris.csv"
lsnb = LooselySymmetricNB()

lsnb.read_csv(path=iris_path)
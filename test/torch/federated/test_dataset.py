import pytest
import torch as th
import syft as sy
from syft.serde import serde
from syft.serde import torch_serde

from syft.frameworks.torch.federated import BaseDataset


def test_base_dataset(workers):

    bob = workers["bob"]
    inputs = th.tensor([1, 2, 3, 4.0])
    targets = th.tensor([1, 2, 3, 4.0])
    dataset = BaseDataset(inputs, targets, id=1, tags=["#test_dataset"], description="test dataset")

    assert len(dataset) == 4
    assert dataset[2] == (3, 3)

    assert dataset.id == 1
    assert dataset.tags[0] == "#test_dataset"
    assert dataset.description == "test dataset"
    dataset.send(bob)
    assert dataset.data.location.id == "bob"
    assert dataset.targets.location.id == "bob"
    assert dataset.location.id == "bob"

    dataset.get()
    with pytest.raises(AttributeError):
        assert dataset.data.location.id == 0
    with pytest.raises(AttributeError):
        assert dataset.targets.location.id == 0


def test_base_dataset_transform():

    inputs = th.tensor([1, 2, 3, 4.0])
    targets = th.tensor([1, 2, 3, 4.0])

    transform_dataset = BaseDataset(inputs, targets)

    def func(x):

        return x * 2

    transform_dataset.transform(func)

    expected_val = th.tensor([2, 4, 6, 8])
    transformed_val = [val[0].item() for val in transform_dataset]

    assert expected_val.equal(th.tensor(transformed_val).long())


def test_federated_dataset(workers):
    bob = workers["bob"]
    alice = workers["alice"]

    alice_base_dataset = BaseDataset(
        th.tensor([3, 4, 5, 6]),
        th.tensor([3, 4, 5, 6]),
        id=1,
        description="alice description",
        tags=["alice's dataset"],
    )
    datasets = [
        BaseDataset(th.tensor([1, 2]), th.tensor([1, 2])).send(bob),
        alice_base_dataset.send(alice),
    ]

    fed_dataset = sy.FederatedDataset(datasets)

    assert fed_dataset.workers == ["bob", "alice"]
    assert len(fed_dataset) == 6

    fed_dataset["alice"].get()
    assert (fed_dataset["alice"].data == alice_base_dataset.data).all()
    assert fed_dataset["alice"][2] == (5, 5)
    assert fed_dataset["alice"].description == "alice description"
    assert fed_dataset["alice"].tags[0] == "alice's dataset"
    assert len(fed_dataset["alice"]) == 4
    assert len(fed_dataset) == 6

    assert isinstance(fed_dataset.__str__(), str)


def test_base_dataset_simplify():
    """This tests our ability to simplify dataset objects
    At the time of writing, dataset  simplify to a tuple where the
    first value in the tuple is the a serialized version of data (tensor) and
    second value is the  dataset's ID
    third value is owner info which is empty
    fourth value is serialized tags for the dataset
    fifth value is serialized description for the dataset
    """

    # create a dataset
    inputs = th.tensor([1, 2, 3, 4.0])
    targets = th.tensor([1, 2, 3, 4.0])
    dataset = BaseDataset(
        inputs, targets, id=10, tags=["#test_dataset"], description="test dataset"
    )

    assert dataset.id == 10
    assert dataset.tags[0] == "#test_dataset"
    assert dataset.description == "test dataset"

    # simplify the dataset
    output = BaseDataset.simplify(dataset)
    assert type(output) == tuple

    # Followed examples from test_serde.py on how it dealt with tensor, strings
    # etc.

    # make sure inner type is correct
    assert type(output[0]) == tuple

    # make sure ID is correctly encoded
    assert output[1] == dataset.id

    # superficially testing data tensor
    assert serde.detailers[output[0][0]] == torch_serde._detail_torch_tensor
    # checking the input tag matches correspodning value  after the tranformation
    # after the tranformation done by the method simplify
    assert output[3][1][0][1][0] == bytes(dataset.tags[0], "utf-8")
    # checking the input description to BaseDataset matches correspodning value
    # after the tranformation done by the method simplify
    assert output[4][1][0] == bytes(dataset.description, "utf-8")


def test_dataset_to_federate(workers):
    bob = workers["bob"]
    alice = workers["alice"]

    dataset = BaseDataset(th.tensor([1.0, 2, 3, 4, 5, 6]), th.tensor([1.0, 2, 3, 4, 5, 6]))

    fed_dataset = dataset.federate((bob, alice))

    assert isinstance(fed_dataset, sy.FederatedDataset)

    assert fed_dataset.workers == ["bob", "alice"]
    assert fed_dataset["bob"].location.id == "bob"
    assert len(fed_dataset) == 6


def test_federated_dataset_search(workers):

    bob = workers["bob"]
    alice = workers["alice"]

    grid = sy.VirtualGrid(*[bob, alice])

    train_bob = th.Tensor(th.zeros(1000, 100)).tag("data").send(bob)
    target_bob = th.Tensor(th.zeros(1000, 100)).tag("target").send(bob)

    train_alice = th.Tensor(th.zeros(1000, 100)).tag("data").send(alice)
    target_alice = th.Tensor(th.zeros(1000, 100)).tag("target").send(alice)

    data, _ = grid.search("data")
    target, _ = grid.search("target")

    datasets = [
        BaseDataset(data["bob"][0], target["bob"][0]),
        BaseDataset(data["alice"][0], target["alice"][0]),
    ]

    fed_dataset = sy.FederatedDataset(datasets)
    train_loader = sy.FederatedDataLoader(fed_dataset, batch_size=4, shuffle=False, drop_last=False)

    counter = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        counter += 1

    assert counter == len(train_loader), f"{counter} == {len(fed_dataset)}"

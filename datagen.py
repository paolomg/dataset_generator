import os
import random
import shutil
from datetime import datetime, timedelta

import numpy as np
from tqdm import tqdm


def random_date(start, end):
    """Return a random datetime between two datetime objects."""
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = random.randrange(int_delta)
    return start + timedelta(seconds=random_second)


class DatasetGenerator:
    def __init__(self,
                 root_path=None,
                 num_plants=1,
                 num_lines=2,
                 num_machines_per_line=4,
                 num_materials=10,
                 max_materials_qty=10,
                 num_stored_materials=500,
                 num_run_days=50,
                 ts_price_start=datetime(2022, 1, 1),
                 ts_stored_start=datetime(2022, 1, 1, 0),
                 ts_stored_end=datetime(2022, 1, 1, 23),
                 ts_run_start=datetime(2022, 1, 2),
                 sell_prob=0.2,
                 price_prob=0.05,
                 error_prob=0.001
                 ):
        self.root_path = os.getcwd() + "/root" if root_path is None else root_path
        self.num_plants = num_plants
        self.num_lines = num_lines
        self.num_machines_per_line = num_machines_per_line
        self.num_materials = num_materials
        self.max_materials_qty = max_materials_qty
        self.num_stored_materials = num_stored_materials
        self.num_run_days = num_run_days
        self.ts_price_start = ts_price_start
        self.ts_stored_start = ts_stored_start
        self.ts_stored_end = ts_stored_end
        self.ts_run_start = ts_run_start
        self.sell_prob = sell_prob
        self.price_prob = price_prob
        self.error_prob = error_prob

        self.product_materials_dict = dict()
        self.topology_dict = dict()
        self.product_machine_dict = dict()
        self.machine_product_dict = dict()
        self.machine_positions_dict = dict()
        self.to_sell_dict = dict()
        self.to_use_dict = dict()
        self.product_time_dict = dict()
        self.machine_status_dict = dict()
        self.machine_working_time_dict = dict()
        self.machine_cycle_time_dict = dict()
        self.orders_dict = dict()
        self.item_idx = 0

        print(f"Warning: generate will remove everything from the "
              f"root_path directory in order to have a new and clean "
              f"dataset. The actual root_path is '{self.root_path}'.")

    def generate(self):
        # Clean the dataset folder when generating a new
        # dataset
        if os.path.exists(self.root_path) and os.path.isdir(self.root_path):
            shutil.rmtree(self.root_path)
        os.makedirs(self.root_path, exist_ok=True)

        self._define_topology()
        self._assign_materials_to_products()
        self._write_bom()
        self._write_materials_costs()
        self._add_materials_from_suppliers()
        self._add_headers()
        self._run_plants()

        """TODO: should probably remove last newline."""

    def _define_topology(self):
        """Defines a topology dictionary of the form:
            {
                plant_name: {
                    line_name: [machines_list]
                }
            }

        The order of the machines inside each line is
        randomized.

        Here we also assign to each machine the product it
        produces and to each item the time required to create
        it. This is a random time between 5 and 20 hours. This
        information is stored in self.product_time_dict.

        Products have IDs that start with 'fg', for
        finished goods.

        We also define three other dictionaries:
            self.product_machine_dict: {product_name: machine_name}
            self.machine_product_dict: {machine_name: product_name}
            self.machine_positions_dict:
                {machine_name: position in machines_list}

        """
        line_idx = 0
        machine_idx = 0
        product_idx = 0

        for plant_idx in range(self.num_plants):
            plant_name = f"plant{plant_idx}"
            self.topology_dict[plant_name] = dict()

            for _ in range(self.num_lines):
                line_name = f"line{line_idx}"
                line_idx += 1
                machines_ls = []

                for _ in range(self.num_machines_per_line):
                    machines_ls.append(machine_idx)
                    machine_idx += 1

                machines_ls = np.random.permutation(machines_ls)
                machines_ls = [f"machine{idx}" for idx in machines_ls]
                self.topology_dict[plant_name][line_name] = machines_ls

                for position, machine_name in enumerate(machines_ls):
                    product_name = f"fg{product_idx}"
                    product_idx += 1

                    self.product_machine_dict[product_name] = machine_name
                    self.machine_product_dict[machine_name] = product_name
                    self.machine_positions_dict[machine_name] = (position, machines_ls)

                    product_time = timedelta(
                        seconds=60 * 60 * np.random.randint(5, 20)
                    )
                    self.product_time_dict[product_name] = product_time

                    self.machine_status_dict[machine_name] = None

                    os.makedirs(
                        f"{self.root_path}/{plant_name}/{line_name}/{machine_name}",
                        exist_ok=True
                    )

    def _assign_materials_to_products(self):
        """Defines a product_material dictionary that
        associates each product with the materials
        required to produce it.

        Every machine needs the product of the previous machine in
        the line (only one product, to avoid stalling the plant due
        to lack of products). The first machine can only use raw
        materials.

        Raw materials have IDs that start with 'rm'.
        """

        for product_name, machine_name in self.product_machine_dict.items():
            machine_pos, machines_ls = self.machine_positions_dict[machine_name]
            materials_ls = []
            if machine_pos > 0:
                previous_machine = machines_ls[machine_pos - 1]
                materials_ls.append(
                    (self.machine_product_dict[previous_machine], 1)
                )

            num_rnd = np.random.exponential(scale=3)
            num_materials_per_product = np.clip(
                int(np.round(num_rnd)), 1, self.num_materials
            )

            # We need to avoid choosing the same material more
            # than once
            chosen_materials = np.random.choice(
                self.num_materials,
                num_materials_per_product,
                replace=False  # Unique values
            )

            for material_idx in chosen_materials:
                materials_ls.append(
                    (f"rm{material_idx}", self._get_random_qty())
                )

            self.product_materials_dict[product_name] = materials_ls

    def _write_bom(self):
        """Writes the BOM file, which lists the materials needed
        to produce each finished item.
        """
        file_path = self.root_path + "/ERP_BOM.csv"
        with open(file_path, "w") as file:
            file.write("FinishGoodID,RawMaterialID,Quantity\n")
            for product, materials_ls in self.product_materials_dict.items():
                for material in materials_ls:
                    file.write(product + ",")
                    file.write(",".join([str(item) for item in material]))
                    file.write("\n")

    def _write_materials_costs(self):
        """Writes the raw material registry file, containing
        the prices of the raw materials. Here we set the prices
        on the day specified in the self.ts_price_start
        variable.

        The price is randomly chosen between 1 and 150.
        """
        file_path = self.root_path + "/ERP_raw_material_registry.csv"
        with open(file_path, "w") as file:
            file.write("RawMaterialID,Price,TS\n")
            for material_idx in range(self.num_materials):
                material_name = f"rm{material_idx}"
                ts = self.ts_price_start
                price = random.randint(1, 150)
                file.write(f"{material_name},{price},{ts}")
                file.write("\n")

    def _add_materials_from_suppliers(self):
        """Add an initial set of raw materials from suppliers,
        with timestamps between self.ts_stored_start and
        self.ts_stored_end.

        We also set up a to_use and a to_sell dictionaries. The
        first contains raw materials and finished goods produced
        by all expect the last machine in a line. The second
        contains the items produced by the last machine in a line.
        We assume that only these last products are going to be
        sold.

        Note that in the 'material_movement' csv we also add am
        ItemTypeID, not present in the exercise description.
        Otherwise, we have no way to know which raw materials
        need to be fetched from storage to produce a new item.
        """
        for plant_name in self.topology_dict:
            self.to_use_dict.setdefault(plant_name, dict())
            self.to_sell_dict.setdefault(plant_name, [])

            file_path = f"{self.root_path}/{plant_name}/" \
                        + f"ERP_material_movement_{plant_name}.csv"
            with open(file_path, "w") as file:
                file.write("ItemID,ItemTypeID,Timestamp,Position\n")

                chosen_materials = np.random.choice(
                    self.num_materials,
                    self.num_stored_materials
                )
                for chosen_material in chosen_materials:
                    material_name = f"rm{chosen_material}"
                    date = random_date(self.ts_stored_start, self.ts_stored_end)
                    file.write(f"{self.item_idx},{material_name},{date},"
                               + "material received from supplier\n")

                    self.to_use_dict[plant_name] \
                        .setdefault(material_name, []).append(
                        (self.item_idx, material_name, date)
                    )

                    self.item_idx += 1

    def _add_headers(self):
        """Add headers to the remaining csv files."""
        for plant_name, lines in self.topology_dict.items():
            for line_name, machines_ls in lines.items():
                status_path = f"{self.root_path}/{plant_name}/{line_name}/" \
                              + f"MES_machine_status_{line_name}.csv"
                with open(status_path, "a") as status_file:
                    status_file.write("Timestamp,MachineID,Status\n")
                for machine_name in machines_ls:
                    cycle_path = f"{self.root_path}/{plant_name}/{line_name}/" \
                                 + f"{machine_name}/MES_cycle_time_{machine_name}.csv"
                    with open(cycle_path, "a") as cycle_file:
                        cycle_file.write("Timestamp,ItemID,ItemTypeID,Cycle Time,Status\n")

    def _run_plants(self):
        """Run all the plants for the number of days
        specified in the self.num_run_days variable.

        After each shift, we order the new raw materials
        needed, if any. After each day, we sell 3/4 of
        the products with self.sell_prob probability.

        The price of a random raw material may change
        each day with self.price_prob probability.
        """
        for day_num in tqdm(
            range(self.num_run_days),
            total=self.num_run_days,
            desc="Running plants"
        ):
            rnd = 0
            time = None
            for plant_name, lines in self.topology_dict.items():
                for line_name, machines_ls in lines.items():
                    for machine_name in machines_ls:
                        for shift_start in [0, 8, 16]:
                            ts_start = self.ts_run_start \
                                       + timedelta(days=day_num) \
                                       + timedelta(hours=shift_start)
                            self._simulate_shift(
                                machine_name,
                                line_name,
                                plant_name,
                                ts_start
                            )
                            self._process_orders(
                                plant_name,
                                ts_start + timedelta(hours=8)
                            )

                time = self.ts_run_start + timedelta(days=day_num + 1)

                # 20% chance to sell products every day
                rnd = np.random.random()
                if rnd <= self.sell_prob:
                    self._sell_products(plant_name, time)

            # 5% chance to change the price of a random
            # raw material
            if rnd <= self.price_prob:
                self._change_price(time)

    def _get_random_qty(self):
        """Generate a random quantity for raw materials,
        using an exponential distribution, so that low
        values are more probable.
        """
        qty_rnd = np.random.exponential(scale=3)
        material_qty = np.clip(
            int(np.round(qty_rnd)), 1, self.max_materials_qty
        )
        return material_qty

    def _store_item(self, machine_name, product_name, plant_name, date, position):
        file_path = f"{self.root_path}/{plant_name}/" \
                    + f"ERP_material_movement_{plant_name}.csv"

        with open(file_path, "a") as file:
            file.write(f"{self.item_idx},{product_name},{date},{position}\n")

        machine_pos, machines_ls = self.machine_positions_dict[machine_name]
        if machine_pos == len(machines_ls) - 1:
            self.to_sell_dict[plant_name].append(
                (self.item_idx, product_name, date)
            )
        else:
            self.to_use_dict[plant_name] \
                .setdefault(product_name, []).append(
                (self.item_idx, product_name, date)
            )

    def _simulate_shift(self, machine_name, line_name, plant_name, ts_start):
        """Simulate a shift of a given machine. The shift is managed with 5
        minutes increments: every 5 minutes we decide what to do and what is
        happening.

        When we run shifts for the first time, we start from the 'Waiting
        for Component' status.

        There is no 'Waiting' status, since I'm not sure what its purpose is.
        With 'Error', we waste the materials used up to now and we go back
        to 'Waiting for Component' with 50% chance. With 'Stop' we wait for
        an order for new materials to be processed.

        Materials are consumed when we start of the working phase, instead
        of after completing the last product of the line, to make restocking
        easier to handle.

        Cycle time is machine time, as far as I could understand.
        """

        status_path = f"{self.root_path}/{plant_name}/{line_name}/" \
                      + f"MES_machine_status_{line_name}.csv"
        cycle_path = f"{self.root_path}/{plant_name}/{line_name}/" \
                     + f"{machine_name}/MES_cycle_time_{machine_name}.csv"

        product_name = self.machine_product_dict[machine_name]
        product_time = self.product_time_dict[product_name]
        materials_required = self.product_materials_dict[product_name]

        # We accumulate working and cycle times in dictionaries,
        # so we can use them even after switching to another machine
        self.machine_working_time_dict.setdefault(
            machine_name, timedelta(seconds=0)
        )
        self.machine_cycle_time_dict.setdefault(
            machine_name, timedelta(seconds=0)
        )

        # 8 hours divided into 5 minutes deltas
        delta_ls = np.arange(0, 60 * 60 * 8, 60 * 5)

        curr_status = self.machine_status_dict[machine_name]
        status_before_break = self.machine_status_dict[machine_name]

        with open(status_path, "a") as status_file:
            for delta in delta_ls:
                self._increase_cycle_time(machine_name)
                time = ts_start + timedelta(seconds=int(delta))

                # Break after 2 hours, 4.15 hours, 7 hours
                if delta == 7200 or delta == 15300 or delta == 25200:
                    status_before_break = curr_status
                    curr_status = "Break"
                    self._write_status(
                        curr_status, machine_name, time, status_file
                    )
                    continue
                # 15 minutes break
                elif (7200 < delta < 8100) or (25200 < delta < 26100):
                    continue
                # 45 minutes break
                elif 15300 < delta < 18000:
                    continue
                # Break finished
                elif delta == 8100 or delta == 18000 or delta == 26100:
                    curr_status = status_before_break
                    self._write_status(
                        curr_status, machine_name, time, status_file
                    )
                    continue

                if curr_status is None:
                    curr_status = "Waiting for Component"
                    self._write_status(
                        curr_status, machine_name, time, status_file
                    )
                elif curr_status == "Waiting for Component":

                    # Remove items. If not enough items, stop
                    ok_flag = self._consume_materials(
                        materials_required, plant_name
                    )
                    if ok_flag:
                        curr_status = "Working"
                        self._write_status(
                            curr_status, machine_name, time, status_file
                        )
                    else:
                        curr_status = "Stop"
                        self._write_status(
                            curr_status, machine_name, time, status_file
                        )
                    # In any case, place order, processed next shift
                    self._place_order(materials_required, plant_name)

                elif curr_status == "Stop":
                    ok_flag = self._consume_materials(
                        materials_required, plant_name
                    )
                    if ok_flag:
                        curr_status = "Working"
                        self._write_status(
                            curr_status, machine_name, time, status_file
                        )

                elif curr_status == "Working":
                    self._increase_working_time(machine_name)

                    if self.machine_working_time_dict[machine_name] \
                            >= product_time:
                        self._create_item(
                            product_name,
                            machine_name,
                            plant_name,
                            time,
                            self.machine_cycle_time_dict[machine_name],
                            cycle_path
                        )
                        self._place_order(materials_required, plant_name)
                        self.machine_working_time_dict[machine_name] = \
                            timedelta(seconds=0)
                        self.machine_cycle_time_dict[machine_name] = \
                            timedelta(seconds=0)

                    # 0.1% chance to have an error by default
                    rnd = np.random.random()
                    if rnd <= self.error_prob:
                        curr_status = "Error"
                        self._write_status(
                            curr_status, machine_name, time, status_file
                        )

                elif curr_status == "Error":

                    # 50% chance to work again
                    rnd = np.random.random()
                    if rnd <= 0.5:
                        # Waiting for Component instead of Working,
                        # so we waste the materials used up to now
                        curr_status = "Waiting for Component"
                        self._write_status(
                            curr_status, machine_name, time, status_file
                        )

                else:
                    raise ValueError(f"Wrong curr_status: {curr_status}, {delta}")

        self.machine_status_dict[machine_name] = curr_status

    def _increase_cycle_time(self, machine_name):
        self.machine_cycle_time_dict[machine_name] += \
            timedelta(seconds=60 * 5)

    def _increase_working_time(self, machine_name):
        self.machine_working_time_dict[machine_name] += \
            timedelta(seconds=60 * 5)

    @staticmethod
    def _write_status(curr_status, machine_name, time, status_file):
        status_file.write(f"{time},{machine_name},{curr_status}\n")

    def _create_item(
            self, product_name, machine_name,
            plant_name, time, cycle_time, cycle_path
    ):
        """Creates the product of a given machine. 5% chance to
        have a 'KO' product. In that case, we don't save the item in
        the material_movement csv, but only in the cycle csv.
        """
        with open(cycle_path, "a") as cycle_file:
            rnd = np.random.random()
            position = "material created"

            # 5% chance to have a broken product
            if rnd <= 0.05:
                status = "KO"
            else:
                status = "OK"
                self._store_item(machine_name, product_name, plant_name, time, position)

            cycle_file.write(
                f"{time},{self.item_idx},{product_name},{cycle_time},{status}\n"
            )

        self.item_idx += 1

    def _place_order(self, materials_required, plant_name):
        self.orders_dict.setdefault(plant_name, []).extend(materials_required)

    def _process_orders(self, plant_name, time):
        """Take the list of required raw materials and
        add them to the to_use_dict, also writing on the
        material_movement csv.
        """
        file_path = f"{self.root_path}/{plant_name}/" \
                    + f"ERP_material_movement_{plant_name}.csv"

        with open(file_path, "a") as file:
            materials_ls = self.orders_dict[plant_name]
            for elem in materials_ls:
                material_name, qty = elem

                # We cannot add finished goods. Those need
                # to be created by machines.
                if material_name.startswith("fg"):
                    continue
                for _ in range(qty):
                    self.to_use_dict[plant_name] \
                        .setdefault(material_name, []).append(
                        (self.item_idx, material_name, time)
                    )
                    file.write(f"{self.item_idx},{material_name},{time},"
                               + "material received from supplier\n")

                    self.item_idx += 1

        # Empty the order list when finished
        self.orders_dict[plant_name] = []

    def _consume_materials(self, materials_required, plant_name):
        """Take the required materials to create a product, updating
        files and dictionaries.

        We return True if everything went fine. If there weren't enough
        materials, we return False, so we can stall the machine until
        an order for new materials is processed.
        """
        new_plant_dict = self.to_use_dict[plant_name].copy()
        res_ls = []

        for elem in materials_required:
            material_name, qty = elem
            for _ in range(qty):
                if material_name not in new_plant_dict or \
                        len(new_plant_dict[material_name]) == 0:
                    return False
                else:
                    res_ls.append(new_plant_dict[material_name][0])
                    new_plant_dict[material_name] = new_plant_dict[material_name][1:]

        file_path = f"{self.root_path}/{plant_name}/" \
                    + f"ERP_material_movement_{plant_name}.csv"

        with open(file_path, "a") as file:
            for elem in res_ls:
                item_idx, material_name, time = elem
                file.write(f"{item_idx},{material_name},{time},"
                           + "material consumed\n")

        self.to_use_dict[plant_name] = new_plant_dict
        return True

    def _change_price(self, time):
        file_path = self.root_path + "/ERP_raw_material_registry.csv"
        with open(file_path, "a") as file:
            material_idx = random.randint(0, self.num_materials)
            material_name = f"rm{material_idx}"
            price = random.randint(1, 150)
            file.write(f"{material_name},{price},{time}\n")

    def _sell_products(self, plant_name, time):
        """Sell roughly 3/4 of the products of the plant."""

        new_to_sell_dict = dict()

        for plant, item_list in self.to_sell_dict.items():
            if plant != plant_name:
                continue

            new_item_list = []
            file_path = f"{self.root_path}/{plant_name}/" \
                        + f"ERP_material_movement_{plant_name}.csv"
            with open(file_path, "a") as file:

                for item in item_list:
                    rnd = np.random.random()
                    if rnd <= 0.75:
                        file.write(f"{item[0]},{item[1]},{time},"
                                   + "material sent to customer\n")
                    else:
                        new_item_list.append(item)

            new_to_sell_dict[plant_name] = new_item_list

        self.to_sell_dict = new_to_sell_dict

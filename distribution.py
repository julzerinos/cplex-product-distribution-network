# Required cplex setup
#   Run `python ./setup.py install` in the
#   cplex/python application installation folder
# More details https://www.ibm.com/docs/en/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html # noqa: E501

import json
from re import M
from docplex.mp.model import Model
from docplex.mp.solution import SolveSolution
from docplex.mp.linear import LinearExpr


# Step 0 - Prepare the environment
def create_model() -> Model:
    return Model('model_distribution')


# Step 1 - Prepare the data
def prepare_data(model: Model, objects_path='.') -> LinearExpr:

    # Data and sets ##

    with open(objects_path + "/model.json", "r") as f:
        modelObjects = json.load(f)

    products = modelObjects['products']

    factories = modelObjects['factories']
    warehouses = modelObjects['warehouses']
    stores = modelObjects['stores']

    truck_types = modelObjects['trucks']
    routes = modelObjects['routes']

    all_points = {}
    all_points.update(factories)
    all_points.update(warehouses)
    all_points.update(stores)

    point_edge_lookup = {
        pname:
            {
                'in': [rname for rname, route in routes.items()
                       if route['end'] == pname],
                'out': [rname for rname, route in routes.items()
                        if route['start'] == pname],
            }
            for pname in all_points
    }

    truck_route_combinations = {
        t + r: {
            'route': r, 'truck': t
        } for t in truck_types for r in routes
        if r in truck_types[t]['possibleRoutes']
    }

    product_route_combinations = {
        p + r: {'route': r, 'product': p} for p in products for r in routes
    }

    warehouse_tier_combinations = {
        w + i: 0 for i, w in enumerate(warehouses)
    }

    # Decision variables ##

    product_quantities = model.continuous_var_dict(
        product_route_combinations, name="dvar_products_per_route"
    )

    truck_quantities = model.integer_var_dict(
        truck_route_combinations, name="dvar_trucks_per_route"
    )

    warehouse_upgrade = model.binary_var_list(
        warehouse_tier_combinations
    )

    # Expressions ##

    # Cost of truck usage
    trucks_rent_cost = model.sum(
        amount * truck_types[truck_route_combinations[tr]['truck']]['dayCost']
        for tr, amount in truck_quantities.items()
    )

    # Cost of product transport
    product_transport_cost = model.sum(
        amount * routes[product_route_combinations[pr]['route']]['costPerUnit']
        for pr, amount in product_quantities.items()
    )

    warehouse_usage_cost_functions = {
        wname: model.piecewise(
            -1,
            [
                (tier['capacity'], tier['cost']) for tier in warehouse['tiers']
            ],
            0
        )
        for wname, warehouse in warehouses.items()
    }

    for _, f in warehouse_usage_cost_functions.items():
        f.plot()

    # Cost of warehouse usage
    warehouse_usage_cost = model.sum(
        warehouse_usage_cost_functions[w](
            model.sum(
                product_quantities[prname] for prname, pr in product_route_combinations.items()
                if pr['route'] in point_edge_lookup[w]['in']
            )
        ) for w in warehouses
    )

    # Constraints ##

    # Routes from factories can have at most the output of the factory
    #   for each product
    for fname, point in factories.items():
        for pname in products:
            model.add_constraint(
                model.sum(
                    product_quantities[pname + r] for r in point_edge_lookup[fname]['out']
                ) <= point['productOutputs'][pname],
                ctname="ct_factories_max_" + pname
            )

    # Sum of routes to the stores must have
    #   at least the demand of the point for each product
    for sname, point in stores.items():
        for pname in products:
            model.add_constraint(
                model.sum(
                    product_quantities[pname + r] for r in point_edge_lookup[sname]['in']
                ) >= point['productInputs'][pname],
                ctname="ct_stores_min_" + sname + pname
            )

    # Sum of truck capacity must be at least the amount of products
    #   transported on its route
    for r in routes:
        model.add_constraint(
            model.sum(
                amount *
                truck_types[truck_route_combinations[tr]
                            ['truck']]['capacity']
                for tr, amount in truck_quantities.items()
                if r == truck_route_combinations[tr]['route']
            ) >= model.sum(
                amount
                for pr, amount in product_quantities.items()
                if r == product_route_combinations[pr]['route']
            ),
            ctname='ct_truck_capacity_at_least_' + r
        )

    # For warehouse points, incoming products should be equal to outgoing products
    for p in products:
        for w in warehouses:
            warehouse_routes = point_edge_lookup[w]
            model.add_constraint(
                model.sum(
                    product_quantities[prname] for prname, pr in product_route_combinations.items()
                    if pr['route'] in warehouse_routes['in']
                    and pr['product'] == p
                )
                ==
                model.sum(
                    product_quantities[prname] for prname, pr in product_route_combinations.items()
                    if pr['route'] in warehouse_routes['out']
                    and pr['product'] == p
                ),
                ctname='ct_warehouse_equilibrium_' + w + p
            )

    # For warehouse points, incoming products can be at most equal to max capacity and at least 0

    for wname, warehouse in warehouses.items():
        model.add_constraint(
            model.sum(
                product_quantities[prname] for prname, pr in product_route_combinations.items()
                if pr['route'] in point_edge_lookup[wname]['in']
            )
            <= warehouse['tiers'][-1]['capacity'],
            ctname='ct_warehouse_max_capacity_' + wname
        )

    for wname, warehouse in warehouses.items():
        model.add_constraint(
            0 <=
            model.sum(
                product_quantities[prname] for prname, pr in product_route_combinations.items()
                if pr['route'] in point_edge_lookup[wname]['in']
            ),
            ctname='ct_warehouse_min_capacity_' + wname
        )

    # Objective function: cost of the distribution ##

    f = trucks_rent_cost + product_transport_cost + warehouse_usage_cost

    return f


# Step 3 - Set the objective
def set_objective(model: Model, objective_function: LinearExpr):
    model.set_objective("min", objective_function)


# Step 4 - Solve the problem
def solve_problem(model: Model) -> SolveSolution:
    solution = model.solve(log_output=True)
    return solution


# Step 5 - Communicate the results
def print_solution(model: Model):
    model.print_information()
    model.print_solution(print_zeros=True)


def save_solution(solution: SolveSolution, test_path='.'):
    with open(test_path + "/solution.json", 'w') as f:
        f.write(solution.export_as_json_string())

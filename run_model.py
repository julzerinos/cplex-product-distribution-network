from distribution import \
    create_model, prepare_data, print_solution, \
    save_solution, set_objective, solve_problem


def main():
    model_objects_path = 'tests/project2'

    model = create_model()

    objective_function = prepare_data(model, model_objects_path)
    set_objective(model, objective_function)

    solution = solve_problem(model)
    if solution:
        print_solution(model)
        save_solution(solution, test_path=model_objects_path)
    else:
        print("No solution found for this model.")

if __name__ == '__main__':
    main()
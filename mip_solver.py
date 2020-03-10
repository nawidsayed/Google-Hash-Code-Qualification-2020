import numpy as np
import pyomo.kernel as pmo
import pyomo.environ

class MIP_solver():
    def __init__(self, simulation):
        self.number = simulation.number
        self.silent = simulation.silent
        self.num_books = simulation.num_books
        self.num_libs = simulation.num_libs
        self.num_days = simulation.num_days
        self.book_points = simulation.book_points
        self.lib_num_books = simulation.lib_num_books
        self.lib_days = simulation.lib_days
        self.lib_ships = simulation.lib_ships
        self.lib_books_lists = simulation.lib_books_lists
        self.lib_books_sets = simulation.lib_books_sets
        self.book_num_libs = simulation.book_num_libs
        self.book_libs_lists = simulation.book_libs_lists

    def get_solver(self, solverName):
        solver = pmo.SolverFactory(solverName)
        if solverName == "cbc":
            solver.options["threads"] = 4
            solver.options["sec"] = 300
        return solver

    def get_optimal_books_for_ordered_libs(self, ind_libs_best, days_available=None, book_points_available=None, solverName="scip"):
        if days_available is None:
            days_available = self.num_days
        if book_points_available is None:
            book_points_available = self.book_points
        lib_num_books_available = np.zeros(self.num_libs, dtype=int)
        ind_libs_available = []
        for lib in ind_libs_best:
            days_available -= self.lib_days[lib]
            lib_num_books_available[lib] = days_available * self.lib_ships[lib]
        ind_libs_available = ind_libs_best
            
        ind_books_available = set()
        for lib in ind_libs_available:
            books = self.lib_books_lists[lib]
            for book in books:
                if book_points_available[book] != 0:
                    ind_books_available.add(book)
        ind_books_available = np.array(list(ind_books_available)) 
        
        book_libs_lists_available = [[] for _ in range(self.num_books)]
        for lib in ind_libs_available:
            books = self.lib_books_lists[lib]
            for book in books:
                book_libs_lists_available[book].append(lib)
        
        self.model = pmo.block()
            
        self.model.books = pmo.variable_dict()
        for book in ind_books_available:
            self.model.books[book] = pmo.variable(lb=0, ub=1)
            
        self.model.lib_books = pmo.variable_dict()
        for lib in ind_libs_available:
            for book in self.lib_books_lists[lib]:
                self.model.lib_books[lib, book] = pmo.variable(domain = pmo.Binary)
                
        self.model.every_book_once = pmo.constraint_list()        
        for book in ind_books_available:
            libs = book_libs_lists_available[book]
            self.model.every_book_once.append(pmo.constraint(sum(self.model.lib_books[lib,book] for lib in libs) <= 1))  
            
        self.model.use_books = pmo.constraint_list()
        for book in ind_books_available:
            libs = book_libs_lists_available[book]
            self.model.use_books.append(pmo.constraint(self.model.books[book] <= sum(self.model.lib_books[lib,book] for lib in libs)))  
            
        self.model.max_books_per_lib = pmo.constraint_list()
        for lib in ind_libs_available:
            books = self.lib_books_lists[lib]
            num_books = lib_num_books_available[lib]
            self.model.max_books_per_lib.append(pmo.constraint(sum(self.model.lib_books[lib, book] for book in books) <= num_books))
            
        self.model.objective = pmo.objective(sum(book_points_available[book] * self.model.books[book] for book in ind_books_available), sense=-1)         

        solver = self.get_solver(solverName)
        solver_result = solver.solve(self.model)  
        
        # Can be used to see solver results
        # print(solver_result)
        
        result = []
        for lib in ind_libs_available:
            books = self.lib_books_lists[lib]
            result.append((lib, [book for book in books if self.model.lib_books.get((lib, book)) != 0]))
            
        return result
    
    def get_best_libs_based_on_remaining_libs(self, ind_libs_available, book_points_available, days_available, solverName="scip"):        
        ind_books_available = set()
        for lib in ind_libs_available:
            books = self.lib_books_lists[lib]
            for book in books:
                ind_books_available.add(book)
        ind_books_available = np.array(list(ind_books_available))
        
        book_libs_lists_available = [[] for _ in range(self.num_books)]
        for lib in ind_libs_available:
            books = self.lib_books_lists[lib]
            for book in books:
                book_libs_lists_available[book].append(lib)
        
        self.model = pmo.block()
            
        self.model.books = pmo.variable_dict()
        for book in ind_books_available:
            self.model.books[book] = pmo.variable(lb=0, ub=1)
            
        self.model.libs = pmo.variable_dict()
        for lib in ind_libs_available:
            self.model.libs[lib] = pmo.variable(domain=pmo.Binary)
                
        self.model.use_books = pmo.constraint_list()
        for book in ind_books_available:
            libs = book_libs_lists_available[book]
            self.model.use_books.append(pmo.constraint(self.model.books[book] <= sum(self.model.libs[lib] for lib in libs)))  
            
        self.model.max_libs = pmo.constraint_list()
        self.model.max_libs.append(pmo.constraint(sum(self.model.libs[lib] * self.lib_days[lib] for lib in ind_libs_available) <= days_available))
            
        self.model.objective = pmo.objective(sum(self.book_points[book] * self.model.books[book] for book in ind_books_available), sense=-1)
        
        solver = self.get_solver(solverName)
        solver_result = solver.solve(self.model)  
        
        result = []
        for lib in ind_libs_available:
            if self.model.libs[lib] != 0:
                result.append(lib)
        return result

    def get_best_libs_unlimited_ships(self, ind_libs_available, days_available, solverName="scip"):
        ind_books_available = set()
        for lib in ind_libs_available:
            books = self.lib_books_lists[lib]
            for book in books:
                ind_books_available.add(book)
        ind_books_available = np.array(list(ind_books_available))
        
        book_libs_lists_available = [[] for _ in range(self.num_books)]
        for lib in ind_libs_available:
            books = self.lib_books_lists[lib]
            for book in books:
                book_libs_lists_available[book].append(lib)
            
        self.model = pmo.block()
            
        self.model.books = pmo.variable_dict()
        for book in ind_books_available:
            self.model.books[book] = pmo.variable(lb=0, ub=1)
            
        self.model.libs = pmo.variable_dict()
        for lib in ind_libs_available:
            self.model.libs[lib] = pmo.variable(domain=pmo.Binary)
                
        self.model.max_libs_in_time = pmo.constraint_list()
        self.model.max_libs_in_time.append(pmo.constraint(sum([self.model.libs[lib] * self.lib_days[lib] for lib in ind_libs_available]) <= days_available))
            
        self.model.use_books = pmo.constraint_list()
        for book in ind_books_available:
            libs = book_libs_lists_available[book]
            self.model.use_books.append(pmo.constraint(self.model.books[book] <= sum(self.model.libs[lib] for lib in libs)))  
            
        self.model.objective = pmo.objective(sum(self.book_points[book] * self.model.books[book] for book in ind_books_available), sense=-1)         
        

        solver = self.get_solver(solverName)
        solver_result = solver.solve(self.model)  
        
        result = []
        for lib in ind_libs_available:
            if self.model.libs[lib] != 0:
                result.append(lib)
        return result

    def get_optimal_ordering_and_books(self, ind_libs_available, days_available=None, book_points_available=None, solverName="scip"):
        if days_available is None:
            days_available = self.num_days
        if book_points_available is None:
            book_points_available = self.book_points
        ind_books_available = set()
        for lib in ind_libs_available:
            books = self.lib_books_lists[lib]
            for book in books:
                if book_points_available[book] != 0:
                    ind_books_available.add(book)
        ind_books_available = np.array(list(ind_books_available))
        
        book_libs_lists_available = [[] for _ in range(self.num_books)]
        for lib in ind_libs_available:
            books = self.lib_books_lists[lib]
            for book in books:
                book_libs_lists_available[book].append(lib)
            
        ind_index_available = np.arange(len(ind_libs_available))

        self.model = pmo.block()

        self.model.books = pmo.variable_dict()
        for book in ind_books_available:
            self.model.books[book] = pmo.variable(lb=0, ub=1)

        self.model.lib_books = pmo.variable_dict()
        for lib in ind_libs_available:
            for book in self.lib_books_lists[lib]:
                self.model.lib_books[lib, book] = pmo.variable(lb=0, ub=1)
                
        self.model.lib_index = pmo.variable_dict()
        for lib in ind_libs_available:
            for index in ind_index_available:
                self.model.lib_index[lib, index] = pmo.variable(domain = pmo.Binary)
        
        self.model.times_remaining = pmo.variable_dict()
        for time in ind_index_available:
            self.model.times_remaining[time] = pmo.variable(lb=0, ub=self.num_days)
            
        self.model.lib_times_remaining = pmo.variable_dict()
        for lib in ind_libs_available:
            self.model.lib_times_remaining[lib] = pmo.variable(lb=0, ub=2*self.num_days)

        self.model.every_book_once = pmo.constraint_list()        
        for book in ind_books_available:
            libs = book_libs_lists_available[book]
            self.model.every_book_once.append(pmo.constraint(sum(self.model.lib_books[lib,book] for lib in libs) <= 1))  
            
        self.model.use_books = pmo.constraint_list()
        for book in ind_books_available:
            libs = book_libs_lists_available[book]
            self.model.use_books.append(pmo.constraint(self.model.books[book] <= sum(self.model.lib_books[lib,book] for lib in libs)))  
                
        self.model.one_place_per_lib = pmo.constraint_list()
        for lib in ind_libs_available:
            self.model.one_place_per_lib.append(pmo.constraint(sum([self.model.lib_index[lib, index] for index in ind_index_available]) == 1))

        self.model.one_lib_per_place = pmo.constraint_list()
        for index in ind_index_available:
            self.model.one_lib_per_place.append(pmo.constraint(sum([self.model.lib_index[lib, index] for lib in ind_libs_available]) == 1))

        self.model.use_libs = pmo.constraint_list()
        for lib in ind_libs_available:
            books = self.lib_books_lists[lib]
            for book in books:
                self.model.use_libs.append(pmo.constraint(self.model.lib_books[lib, book] <= sum([self.model.lib_index[lib, index] for index in ind_index_available])))
            
        self.model.remaining_times = pmo.constraint_list()
        for time in ind_index_available:
            prev_time = self.model.times_remaining[time - 1] if time else days_available
            self.model.remaining_times.append(pmo.constraint(self.model.times_remaining[time] == prev_time - sum([self.model.lib_index[lib, time] * self.lib_days[lib] for lib in ind_libs_available])))
        
        self.model.lib_remaining_times = pmo.constraint_list()
        for time in ind_index_available:
            for lib in ind_libs_available:
                self.model.lib_remaining_times.append(pmo.constraint(self.model.lib_times_remaining[lib] <= self.model.times_remaining[time] + days_available * (1-self.model.lib_index[lib, time])))
        
        self.model.lib_max_num_books = pmo.constraint_list()
        for lib in ind_libs_available:
            books = self.lib_books_lists[lib]
            self.model.lib_max_num_books.append(pmo.constraint(sum([self.model.lib_books[lib, book] for book in books]) <= self.lib_ships[lib] * self.model.lib_times_remaining[lib]))
      
        self.model.objective = pmo.objective(sum(book_points_available[book] * self.model.books[book] for book in ind_books_available), sense=-1)                     

        solver = self.get_solver(solverName)
        solver_result = solver.solve(self.model)  

        result = []
        for index in range(len(ind_libs_available)):
            for lib in ind_libs_available:
                if self.model.lib_index.get((lib, index)) != 0:
                    books = self.lib_books_lists[lib]
                    result.append((lib, [book for book in books if self.model.lib_books.get((lib, book)) != 0]))
            
        return result



    




























    def get_optimal_ordering_and_books_for_subsection(self, ind_libs_available, lo, hi, solverName="scip"):
        days_available = self.num_days
        book_points_available = self.book_points
        ind_books_available = set()
        for lib in ind_libs_available:
            books = self.lib_books_lists[lib]
            for book in books:
                if book_points_available[book] != 0:
                    ind_books_available.add(book)
        ind_books_available = np.array(list(ind_books_available))
        
        book_libs_lists_available = [[] for _ in range(self.num_books)]
        for lib in ind_libs_available:
            books = self.lib_books_lists[lib]
            for book in books:
                book_libs_lists_available[book].append(lib)
            
        ind_index_available = np.arange(hi-lo)
        ind_libs_available_reorder = ind_libs_available[lo:hi]
        days_available_reorder = days_available - self.lib_days[ind_libs_available[:lo]].sum()

        lib_num_books_available = np.zeros(self.num_libs, dtype=int)
        days_remaining = days_available
        for lib in ind_libs_available:
            days_remaining -= self.lib_days[lib]
            lib_num_books_available[lib] = days_remaining * self.lib_ships[lib]

        self.model = pmo.block()

        self.model.books = pmo.variable_dict()
        for book in ind_books_available:
            self.model.books[book] = pmo.variable(lb=0, ub=1)

        self.model.lib_books = pmo.variable_dict()
        for lib in ind_libs_available:
            for book in self.lib_books_lists[lib]:
                self.model.lib_books[lib, book] = pmo.variable(lb=0, ub=1)
                
        self.model.lib_index = pmo.variable_dict()
        for lib in ind_libs_available_reorder:
            for index in ind_index_available:
                self.model.lib_index[lib, index] = pmo.variable(domain = pmo.Binary)
        
        self.model.times_remaining = pmo.variable_dict()
        for time in ind_index_available:
            self.model.times_remaining[time] = pmo.variable(lb=0, ub=self.num_days)

        self.model.lib_times_remaining = pmo.variable_dict()
        for lib in ind_libs_available_reorder:
            self.model.lib_times_remaining[lib] = pmo.variable(lb=0, ub=2*self.num_days)

        self.model.every_book_once = pmo.constraint_list()        
        for book in ind_books_available:
            libs = book_libs_lists_available[book]
            self.model.every_book_once.append(pmo.constraint(sum(self.model.lib_books[lib,book] for lib in libs) <= 1))  
            
        self.model.use_books = pmo.constraint_list()
        for book in ind_books_available:
            libs = book_libs_lists_available[book]
            self.model.use_books.append(pmo.constraint(self.model.books[book] <= sum(self.model.lib_books[lib,book] for lib in libs)))  

        self.model.one_place_per_lib = pmo.constraint_list()
        for lib in ind_libs_available_reorder:
            self.model.one_place_per_lib.append(pmo.constraint(sum([self.model.lib_index[lib, index] for index in ind_index_available]) == 1))

        self.model.one_lib_per_place = pmo.constraint_list()
        for index in ind_index_available:
            self.model.one_lib_per_place.append(pmo.constraint(sum([self.model.lib_index[lib, index] for lib in ind_libs_available_reorder]) == 1))

        self.model.use_libs = pmo.constraint_list()
        for lib in ind_libs_available_reorder:
            books = self.lib_books_lists[lib]
            for book in books:
                self.model.use_libs.append(pmo.constraint(self.model.lib_books[lib, book] <= sum([self.model.lib_index[lib, index] for index in ind_index_available])))

        self.model.remaining_times = pmo.constraint_list()
        for time in ind_index_available:
            prev_time = self.model.times_remaining[time - 1] if time else days_available_reorder
            self.model.remaining_times.append(pmo.constraint(self.model.times_remaining[time] == prev_time - sum([self.model.lib_index[lib, time] * self.lib_days[lib] for lib in ind_libs_available_reorder])))

        self.model.lib_remaining_times = pmo.constraint_list()
        for time in ind_index_available:
            for lib in ind_libs_available_reorder:
                self.model.lib_remaining_times.append(pmo.constraint(self.model.lib_times_remaining[lib] <= self.model.times_remaining[time] + days_available * (1-self.model.lib_index[lib, time])))
        
        self.model.lib_max_num_books = pmo.constraint_list()
        for lib in ind_libs_available_reorder:
            books = self.lib_books_lists[lib]
            self.model.lib_max_num_books.append(pmo.constraint(sum([self.model.lib_books[lib, book] for book in books]) <= self.lib_ships[lib] * self.model.lib_times_remaining[lib]))

        self.model.max_books_per_lib = pmo.constraint_list()
        for lib in ind_libs_available:
            if not lib in list(ind_libs_available_reorder):
                books = self.lib_books_lists[lib]
                num_books = lib_num_books_available[lib]
                self.model.max_books_per_lib.append(pmo.constraint(sum(self.model.lib_books[lib, book] for book in books) <= num_books))
                
        self.model.objective = pmo.objective(sum(book_points_available[book] * self.model.books[book] for book in ind_books_available), sense=-1)                     

        solver = self.get_solver(solverName)
        solver_result = solver.solve(self.model)  


        # Can be used to see solver results
        # print(solver_result)

        result = []


        for lib in ind_libs_available[:lo]:
            books = self.lib_books_lists[lib]
            result.append((lib, [book for book in books if self.model.lib_books.get((lib, book)) != 0]))

        for index in ind_index_available:
            for lib in ind_libs_available_reorder:
                if self.model.lib_index.get((lib, index)) != 0:
                    books = self.lib_books_lists[lib]
                    result.append((lib, [book for book in books if self.model.lib_books.get((lib, book)) != 0]))
 
        for lib in ind_libs_available[hi:]:
            books = self.lib_books_lists[lib]
            result.append((lib, [book for book in books if self.model.lib_books.get((lib, book)) != 0]))

        return result



    
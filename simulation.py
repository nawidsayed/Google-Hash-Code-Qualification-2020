import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

paths = ['data/a_example.txt', 'data/b_read_on.txt', 'data/c_incunabula.txt', 'data/d_tough_choices.txt', 'data/e_so_many_books.txt', 'data/f_libraries_of_the_world.txt']
length_paths = len(paths)

def write(number, list_lists):
    path = paths[number][:-4]+"_out.txt"
    with open(path, 'w') as f:    
        for lis in list_lists:
            for e in lis[:-1]:
                f.write(str(e) + " ")
            f.write(str(lis[-1]) + "\n")
        f.close()
        
def line_to_ints(line):
    return [int(s) for s in line.split(" ")]  

def read_submission(number):
    path = paths[number][:-4]+"_out.txt"
    solution = []
    with open(path, 'r') as f:    
        num_libs = line_to_ints(f.readline())[0]
        for i in range(num_libs):
            lib, _ = line_to_ints(f.readline())
            books = line_to_ints(f.readline())
            solution.append((lib, books))
        f.close()    
    return solution

def read(number):
    path = paths[number]
    with open(path, 'r') as f:
        num_books, num_libs, num_days = line_to_ints(f.readline())
        book_points = line_to_ints(f.readline())
        lib_books_lists = []
        lib_num_books = []
        lib_days = []
        lib_ships = []
        for lib in range(num_libs):
            lib_num_book, lib_day, lib_ship = line_to_ints(f.readline())
            lib_num_books.append(lib_num_book)
            lib_days.append(lib_day)
            lib_ships.append(lib_ship)
            lib_books_lists.append(line_to_ints(f.readline()))

        book_points = np.array(book_points)
        lib_num_books = np.array(lib_num_books)
        lib_days = np.array(lib_days)
        lib_ships = np.array(lib_ships)
            
    return num_books, num_libs, num_days, book_points, lib_num_books, lib_days, lib_ships, lib_books_lists


class Simulation_Base():
    
    def transform_solution_into_submission(self):
        self.submission = [[0]]
        for lib, books in self.solution:
            if len(books) > 0:
                self.submission[0][0] += 1
                self.submission.append([lib, len(books)])
                self.submission.append(books)
            
    def write(self):
        self.transform_solution_into_submission()
        write(self.number, self.submission)
        
    def read_submission(self):
        self.solution = read_submission(self.number)
        self.check_solution()
        
    def get_score(self):
        set_books = set()
        for lib, books in self.solution:
            for book in books:
                set_books.add(book)
        ind_books = np.array(list(set_books))
        if len(ind_books) > 0:
            return np.sum(self.book_points[ind_books])
        return 0
    
    def check_solution(self):
        libs_used = set()
        days_remaining = self.num_days
        for lib, books in self.solution:
            if lib in libs_used:
                print("fail, lib already used", lib)
                return False
            days_remaining -= self.lib_days[lib]
            if days_remaining < 0:
                print("fail, no days remaining to add lib", days_remaining)
                return False
            if len(set(books).difference(self.lib_books_sets[lib])) > 0:
                print("fail, books are not in bib", set(books).difference(self.lib_books_sets))
                return False
            if len(set(books)) != len(books):
                print("fail, book used multiple times in lib", books)
                return False
            if len(books) > self.lib_ships[lib] * days_remaining:
                print("fail, to many books for this lib", len(books), days_remaining, self.lib_ships[lib])
                return False
        return True
    
    def init_solution(self):
        self.solution = []
        self.lib_books_used = [set() for _ in range(self.num_libs)]
        self.lib_ind_counter = 0
        self.lib_ind = np.zeros(self.num_libs, dtype=int) - 1
        self.time_solution_left = self.num_days
        self.lib_remaining_books_ship = np.zeros(self.num_libs, dtype=int) - 1
        self.libs_used = set()
    
    def add_lib(self, lib):
        if lib in self.libs_used:
            if not self.silent:
                print("flop, lib used:", lib)
            return False
        if self.time_solution_left - self.lib_days[lib] < 0:
            if not self.silent:
                print("flop, no more days left", self.time_solution_left)
            return False
        self.time_solution_left -= self.lib_days[lib]
        self.solution.append((lib, []))
        self.libs_used.add(lib)
        self.lib_ind[lib] = self.lib_ind_counter
        self.lib_ind_counter += 1
        self.lib_remaining_books_ship[lib] = self.time_solution_left * self.lib_ships[lib]
        return True

    def add_lib_book(self, lib, book):
        if not book in self.lib_books_sets[lib]:
            if not self.silent:
                print("flop, book not in lib:", lib, book)
            return False
        if self.lib_remaining_books_ship[lib] < 1:
            if not self.silent:
                print("flop, no more books to ship:", lib, book)
            return False
        if book in self.lib_books_used[lib]:
            if not self.silent:
                print("flop, books already added to lib", lib, book)
        self.solution[self.lib_ind[lib]][1].append(book)
        self.lib_books_used[lib].add(book)
        self.lib_remaining_books_ship[lib] -= 1
        return True

    def get_solution_stats(self):
        lib_stats = []
        day = 0
        for lib, books in self.solution:
            day += self.lib_days[lib]
            time_wasted = (self.num_days - day) - self.lib_num_books[lib] / self.lib_ships[lib] 
            if len(books) <= 0:
                avg = 0
                summ = 0
                mini = 0
                maxi = 0
            else:            
                avg = np.mean(self.book_points[books])
                summ = np.sum(self.book_points[books])
                mini = np.min(self.book_points[books])
                maxi = np.max(self.book_points[books])
            lib_stats.append([day, summ, summ/self.lib_days[lib], avg, mini, maxi, self.lib_days[lib], self.lib_ships[lib], len(books), time_wasted, self.lib_ships[lib] / self.lib_days[lib]])
        columns = ["day", "sum_points", "sum_points_per_day", "avg_points", "min_points", "max_points", "lib_days", "lib_ships", "num_books", "days_wasted", "efficiency"]
        return pd.DataFrame(data=np.array(lib_stats), columns=columns)
    
    def plot_solution(self):
        data = self.get_solution_stats()
        plt.figure(figsize=(10,50))
        num_plots = len(data.columns[1:])
        for i, column in enumerate(data.columns[1:]):
            plt.subplot(num_plots,1,i+1)
            sns.lineplot(data=data, x="day", y=column)
        
        
    def __init__(self, number, silent = False):
        self.number = number
        self.silent = silent
        num_books, num_libs, num_days, book_points, lib_num_books, lib_days, lib_ships, lib_books_lists = read(number)
        
        self.num_books = num_books
        self.num_libs = num_libs
        self.num_days = num_days
        self.book_points = book_points
        self.lib_num_books = lib_num_books
        self.lib_days = lib_days
        self.lib_ships = lib_ships
        self.lib_books_lists = [list(np.array(books)[np.argsort(self.book_points[books])[::-1]]) for books in lib_books_lists]



        self.lib_books_sets = [set(books) for books in self.lib_books_lists]

        self.book_num_libs = np.zeros(self.num_books, dtype = int)        
        self.book_libs_lists = [[] for _ in range(self.num_books)]        
        for lib, books in enumerate(self.lib_books_lists):
            for book in books:
                self.book_libs_lists[book].append(lib)
            self.book_num_libs[books] += 1


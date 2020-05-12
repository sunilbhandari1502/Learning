from Examples import StudentClass as std
from Examples import Inheritancetest as InT

# from . import StudentClass as std # . means same directory as this module
# First Student object is created
stud1 = std.Student(10, "Jack", "MS")  # this call will execute the code in the bottom of the StudentClass.py
# First Student object is created
stud2 = std.Student(30, "Jill", "BE")
# Display the details of first student
stud1.displayStudent()
at = getattr(stud1, 'name')
print("getattr(stud1,'name'): ", at)
print("hasattr(stud1,'age'): ", hasattr(stud1, 'age'))
# New attribute inserted
print("setattr(stud1,age,21): ", setattr(stud1, 'age', 21))
stud1.displayStudent()
print("Age : ", stud1.age)
# Attribute age deleted
print("delattr(stud1,age): ", delattr(stud1, 'age'))
stud1.displayStudent()
# Display the details of Second student
stud2.displayStudent()
print("Total number of students: ", std.Student.studentcount)

# built-in class attributes
# _dict_ : contains the dictionary containing the class's namespace
# _doc_ : contains the class documentation string
# _name_ : contains the class name
# _module_ : contains module name in which class is defined. This attribute is "_main_" in interactive mode
# _bases_ : contains an empty tuple containing the base classes, in the order of their ocurrence in the base class list

print("Student.__doc__: ", std.Student.__doc__)
print("Student.__name__: ", std.Student.__name__)
print("Student.__module__: ", std.Student.__module__)
print("Student.__bases__: ", std.Student.__bases__)
print("Student.__dic__: ", std.Student.__dict__)

# deleting one student object of the Student class
# python automatically deletes an object that is no longer in use Garbage cpllecter..
# in the example it is destroying stud2 automatically
del stud1

r = 50
n = "SunilB"
c = "MFE"
m = 580
# Creating the object
print("Result")
stud3 = InT.Test()
stud3.getDetails(r, n, c)
stud3.getMarks(m)
stud3.displayStudent()
stud3.displayMarks()

# Multiple inheritance  Test inherit Student and Results inherit Test
stud4 = InT.Results()
stud4.getDetails(r, n, c)
stud4.getMarks(m)
stud4.displayStudent()
stud4.displayMarks()
stud4.calculateGrade()

# The Student and class Teacher both have __init__() method.
# The __init__() method is defined in class Student and Teacher is extended in class School
sc = InT.School(10, "Jones", "BE", 100, "Jack", "Math", 80)
sc.displayStudent()
sc.showTeacher()
sc.showSchool()

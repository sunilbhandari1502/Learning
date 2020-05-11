class Student:
    """Common base class for all students"""
    studentcount = 0

    def __init__(self, rollno, name, course):
        self.rollno = rollno
        self.name = name
        self.course = course
        Student.studentcount += 1

    def displayCount(self):
        print("Total Students", Student.studentcount)

    def displayStudent(self):
        print("Roll Number:", self.rollno)
        print("Name: ", self.name)
        print("Course: ", self.course)

    def getDetails(self, rollno, name, course):
        self.rollno = rollno
        self.name = name
        self.course = course

    def __del__(self):
        class_name = self.__class__.__name__
        print(class_name, " object ", self.name, " destroyed")


# First Student object is created
stud1 = Student(20, "John", "ES")
# First Student object is created
stud2 = Student(40, "Tim", "MB")
# Display the details of first student
stud1.displayStudent()
# Display the details of Second student
stud2.displayStudent()
print("Total number of students: ", Student.studentcount)


# # Inheritance
# class Test(Student):
#     def __init__(self):
#         pass
#
#     def getMarks(self, marks):
#         self.marks = marks
#
#     def displayMarks(self):
#         print("Total Marks :", self.marks)


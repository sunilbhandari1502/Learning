from Examples import StudentClass as std


# Inheritance
class Test(std.Student):
    def __init__(self):
        pass

    def getMarks(self, marks):
        self.marks = marks

    def displayMarks(self):
        print("Total Marks :", self.marks)

    def displayStudent(self):
        print("Roll Number:", self.rollno)
        print("Name: ", self.name)
        print("Course: ", self.course)
        print("Calling Overridden method of Child")


class Results(Test):
    # def __init__(self):
    #     pass

    def calculateGrade(self):
        if self.marks > 480:
            self.grade = "Distinction"
        elif self.marks > 360:
            self.grade = "First Class"
        elif self.marks > 240:
            self.grade = "Second Class"
        else:
            self.grade = "Failed"
        print("Result:", self.grade)


class Teacher(object):
    def __init__(self, teac_id, teac_name, subject):
        self.teac_id = teac_id
        self.teac_name = teac_name
        self.subject = subject

    def showTeacher(self):
        print("Id of teacher: ", self.teac_id)
        print("Name of teacher: ", self.teac_name)
        print("Subject: ", self.subject)


# Instead of using function of subclass constructor of the subclass used to pass value to object
class School(std.Student, Teacher):
    def __init__(self, ID, name, course, teac_id, teac_name, subject, sch_id):
        std.Student.__init__(self, ID, name, course)
        Teacher.__init__(self, teac_id, teac_name, subject)
        self.sch_id = sch_id

    def showSchool(self):
        print("Id of the school", self.sch_id)

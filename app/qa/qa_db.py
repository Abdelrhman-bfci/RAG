from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from app.config import Config
from app.models import Institute, Department, Bylaw, Program, Course

def get_session():
    engine = create_engine(Config.DATABASE_URL)
    Session = sessionmaker(bind=engine)
    return Session()

def get_courses_engineering_computer():
    """
    Give me all courses in Faculty of Engineering (Institute) 
    that is provided by computer department (Departments).
    """
    session = get_session()
    try:
        results = session.query(Course, Program, Bylaw, Department, Institute).\
            join(Program, Course.program_id == Program.id).\
            join(Bylaw, Program.bylaw_id == Bylaw.id).\
            join(Department, Bylaw.department_id == Department.id).\
            join(Institute, Department.institute_id == Institute.id).\
            filter(Institute.name.like("%Faculty of Engineering%") or Institute.name.like("%الهندسة%")).\
            filter(Department.name.like("%Computer%") or Department.name.like("%الحاسب%")).\
            all()
        
        courses_list = []
        for course, program, bylaw, department, institute in results:
            courses_list.append({
                "course_name": course.name,
                "bylaw_id": bylaw.id,
                "institute_id": institute.id,
                "department_id": department.id
            })
        return courses_list
    finally:
        session.close()

def count_courses_electrical_engineering():
    """
    Give number of courses provided by electrical department in Faculty of Engineering.
    """
    session = get_session()
    try:
        count = session.query(func.count(Course.id)).\
            join(Program, Course.program_id == Program.id).\
            join(Bylaw, Program.bylaw_id == Bylaw.id).\
            join(Department, Bylaw.department_id == Department.id).\
            join(Institute, Department.institute_id == Institute.id).\
            filter(Institute.name.like("%Faculty of Engineering%") or Institute.name.like("%الهندسة%")).\
            filter(Department.name.like("%Electrical%") or Department.name.like("%الكهرباء%")).\
            scalar()
        return count
    finally:
        session.close()

def get_courses_4_credit_hours():
    """
    Give me the list of courses with 4 credit hours.
    """
    session = get_session()
    try:
        results = session.query(Course).filter(Course.credit_hours == 4).all()
        return [course.name for course in results]
    finally:
        session.close()

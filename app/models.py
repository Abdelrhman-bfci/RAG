from sqlalchemy import Column, Integer, String, ForeignKey, Table
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

class Institute(Base):
    __tablename__ = "institutes"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    parent_id = Column(Integer, ForeignKey("institutes.id"), nullable=True)
    
    children = relationship("Institute", backref="parent", remote_side=[id])
    departments = relationship("Department", back_populates="institute")
    bylaws = relationship("Bylaw", back_populates="institute")

class Department(Base):
    __tablename__ = "departments"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    parent_id = Column(Integer, ForeignKey("departments.id"), nullable=True)
    institute_id = Column(Integer, ForeignKey("institutes.id"), nullable=False)

    children = relationship("Department", backref="parent", remote_side=[id])
    institute = relationship("Institute", back_populates="departments")
    bylaws = relationship("Bylaw", back_populates="department")

class Bylaw(Base):
    __tablename__ = "bylaws"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    department_id = Column(Integer, ForeignKey("departments.id"), nullable=True)
    institute_id = Column(Integer, ForeignKey("institutes.id"), nullable=False)

    institute = relationship("Institute", back_populates="bylaws")
    department = relationship("Department", back_populates="bylaws")
    programs = relationship("Program", back_populates="bylaw")

class Program(Base):
    __tablename__ = "programs"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    bylaw_id = Column(Integer, ForeignKey("bylaws.id"), nullable=False)
    parent_id = Column(Integer, ForeignKey("programs.id"), nullable=True)

    children = relationship("Program", backref="parent", remote_side=[id])
    bylaw = relationship("Bylaw", back_populates="programs")
    courses = relationship("Course", back_populates="program")

class Course(Base):
    __tablename__ = "courses"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    credit_hours = Column(Integer, nullable=False)
    program_id = Column(Integer, ForeignKey("programs.id"), nullable=False)

    program = relationship("Program", back_populates="courses")

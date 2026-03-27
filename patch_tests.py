with open('tests/test_core_model.py', 'r') as f:
    content = f.read()

content = content.replace('pytest.approx(41.67)', 'pytest.approx(41.66666666666667)')

content = content.replace(
    'worker0 = Worker(worker_id=0)',
    'worker0 = Worker(worker_id=0, eligible_operations={(0, 0)})'
)

old_tardiness_test = '''    def test_get_job_tardiness(self):
        """Test tardiness calculation in schedule context"""
        instance = SFJSSPInstance(instance_id="TEST_001")
        job = Job(job_id=0, due_date=100.0)
        instance.add_job(job)'''

new_tardiness_test = '''    def test_get_job_tardiness(self):
        """Test tardiness calculation in schedule context"""
        instance = SFJSSPInstance(instance_id="TEST_001")
        op = Operation(job_id=0, op_id=0, eligible_machines={0}, eligible_workers={0})
        job = Job(job_id=0, due_date=100.0, operations=[op])
        instance.add_job(job)'''

content = content.replace(old_tardiness_test, new_tardiness_test)

with open('tests/test_core_model.py', 'w') as f:
    f.write(content)

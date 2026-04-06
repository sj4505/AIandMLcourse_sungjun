"""PySide6 GUI 초기화 테스트 (pytest-qt)"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def app(qapp):
    """QApplication fixture (pytest-qt 제공)"""
    return qapp


def test_main_window_creates(app):
    """MainWindow가 예외 없이 생성되는지 확인"""
    from week4_app import MainWindow
    win = MainWindow()
    assert win is not None
    win.close()


def test_tab_count(app):
    """탭이 4개인지 확인"""
    from week4_app import MainWindow
    win = MainWindow()
    assert win.centralWidget().count() == 4
    win.close()


def test_tab_titles(app):
    """탭 이름 확인"""
    from week4_app import MainWindow
    win = MainWindow()
    tabs = win.centralWidget()
    titles = [tabs.tabText(i) for i in range(tabs.count())]
    assert "Lab 1" in titles[0]
    assert "Lab 2" in titles[1]
    assert "Lab 3" in titles[2]
    assert "Lab 4" in titles[3]
    win.close()

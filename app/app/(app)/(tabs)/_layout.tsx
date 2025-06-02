import { Tabs } from 'expo-router'
import { EvilIcons } from '@expo/vector-icons';
import BlurredHeader from '../_headers/BlurredHeader';

const ICON_SIZE = 30;

const options = (title: string, tintColor: string, iconName: "comment" | "search" | "gear") => {
    return {
        header: () => <BlurredHeader title={title} />,
        tabBarActiveTintColor: tintColor,
        tabBarLabel: title,
        tabBarIcon: (({ color }: any) => <EvilIcons color={color} name={iconName} size={ICON_SIZE} />),
    }
}

export default function _layout() {
    return <Tabs screenOptions={
        {
            animation: 'shift',
            headerTitleAlign: 'left',
            tabBarStyle: { height: 60, marginBottom: 10 },
            tabBarLabelPosition: 'below-icon',
        }
    }>
        <Tabs.Screen name='index' options={options('chats', 'green', 'comment')} />

        <Tabs.Screen name='explore' options={options('explore', 'red', 'search')} />

        <Tabs.Screen name='settings' options={options('settings', 'yellow', 'gear')} />
    </Tabs>;
}


